# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from packaging import version
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)
from diffusers import (
    StableDiffusionPipeline,
    DiffusionPipeline,
    StableDiffusionMixin,
    ControlNetModel,
)

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models import (
    AutoencoderKL,
    ImageProjection,
    UNet2DConditionModel,
    attention_processor,
)
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor, is_compiled_module
import torch.nn.functional as F

# from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion.pipeline_output import (
    StableDiffusionPipelineOutput,
)
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from math import sqrt
from collections import defaultdict
import numpy as np
import torchvision.transforms.functional as TF

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True
    )
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = (
        guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    )
    return noise_cfg


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def register_attention_control(pipeline, args, default=True, attention_store=None):
    # replace attn_processor with our own to edit activations
    if default:
        processor = attention_processor.AttnProcessor()
        pipeline.unet.set_attn_processor(processor)
    else:
        new_attn_dict = {}
        attn_procs = pipeline.unet.attn_processors
        att_count = 0
        for name in attn_procs.keys():
            att_count += 1
            new_attn_dict[name] = PLProcessor(
                key=name, args=args, attention_store=attention_store
            )
        pipeline.unet.set_attn_processor(new_attn_dict)
        if attention_store is not None:
            attention_store.num_att_layers = att_count


class AttentionStore:
    def __init__(self, args):
        """
        stores attention layer information
        """
        self.cur_layer_step = 0
        self.num_att_layers = -1
        self.save_layer_names = args.inject_layers
        self.valid_step_range = args.inject_t
        self.cur_att_layer = 0
        self.curr_timestep = 0
        self.curr_timestep_index = 0
        self.avg_heads = False
        # self.save_global_store = False
        # self.global_dict = self.get_empty_store()
        self.attention_store = self.get_empty_store()
        # self.global_store = {}
        self.keep_grad = False
        self.step_store = self.get_empty_store()
        self.freeze = False
        self.inject_features = False
        self.first_timestep_index = -1

    def __call__(self, attn, layer_name: str, save_cross: bool):
        """
        will store activations in internal dictionary
        attn: activations (b x head x hxw x f) as torch tensor
        layer_name: which layer they came from (name, in string)
        save_cross: controls whether cross attention or self attention are saved?
        """
        if not self.freeze:
            if self.cur_timestep_in_range():
                self.forward(attn, layer_name, save_cross)
                self.cur_att_layer += 1
                if self.cur_att_layer == self.num_att_layers:
                    self.cur_att_layer = 0
                    self.cur_layer_step += 1
                    # self.between_steps()

    def forward(self, attn, layer_name, save_cross):
        # parse layer name into short info
        location, is_cross = self.parse_layer_name(layer_name)
        shortcut_key = self.key_to_shortcut(layer_name)
        if self.avg_heads:
            # averages over heads & selects only positive prompts
            attn_shape = (
                attn.shape
            )  # (2*b*h, height*width, f), since we call this function during CFG it is 2*b and not b
            attn = attn.reshape(
                -1, 8, attn_shape[1], attn_shape[2]
            )  # (2*b, h, height*width, f)
            attn = attn.mean(dim=1, keepdim=True)  # (2*b,1, hxw,f)
            attn = attn[attn.shape[0] // 2 :, :, :, :]  # (b,1, hxw,f)
            to_save = attn
        else:
            to_save = attn  # (2*b*h, seq, f) where seq=height*width, and f=height*width or max seq len
        if shortcut_key in self.save_layer_names:
            if (save_cross and is_cross) or (not save_cross and not is_cross):
                # print("storing: {}, {}".format(layer_name, to_save.shape))
                if self.keep_grad:
                    self.attention_store[shortcut_key].append(to_save)
                else:
                    self.attention_store[shortcut_key].append(
                        to_save.detach().cpu().numpy()
                    )

    def set_timesstep(self, t, i):
        """
        t is a value between 0 and 1, where 0 is the begining of the *DENOISING* process (fully noisy) and 1.0 is image
        """
        self.curr_timestep = t
        self.curr_timestep_index = i
        # print(self.curr_timestep_index)

    def get_empty_store(self):
        return defaultdict(list)

    def cur_timestep_in_range(self):
        condition = (
            self.valid_step_range[0] <= self.curr_timestep <= self.valid_step_range[1]
        )
        if condition and self.first_timestep_index == -1:
            self.first_timestep_index = self.curr_timestep_index
        return condition

    def get_ss(self, features, head_index=-1):
        if head_index > 0:
            sel_features = features[head_index : head_index + 1]
        else:
            sel_features = features
        ss = torch.linalg.matrix_power(features, 20)[:, 0]
        return ss

    def eigen_decomposition(self, features, head_index=-1):
        assert features.ndim == 3  # (b, seq, seq)
        if head_index > 0:
            sel_features = features[head_index : head_index + 1]
        else:
            sel_features = features
        sel_features = torch.transpose(sel_features, 1, 2)
        eigen_values, eigen_vectors = torch.linalg.eig(sel_features)
        ## following lines edits original vectors !!!
        ss_selector = torch.isclose(
            torch.real(eigen_values), torch.tensor(1.0), rtol=1e-4
        )
        ss = torch.real(
            eigen_vectors.permute(0, 2, 1)[ss_selector]
        )  # (b, h*w) with norm = 1
        ss[(ss < 0).any(dim=1)] *= -1  # use a positive normalized vector
        assert ss.all().item()
        return ss, ss_selector, eigen_vectors, eigen_values

    def eigen_reconstruction(self, ss, ss_selector, eigen_vectors, eigen_values):
        eigen_vectors.permute(0, 2, 1)[ss_selector] = ss.to(torch.complex64)
        values = torch.diag_embed(eigen_values)
        t_vectors = torch.transpose(eigen_vectors, 1, 2)
        t_values = torch.transpose(values, 1, 2)
        res = torch.linalg.solve(t_vectors, t_values)
        reconst = eigen_vectors @ torch.transpose(res, 1, 2)
        reconst = torch.real(reconst)
        return reconst.permute(0, 2, 1)

    def edit_ss(self, features, head=-1):
        spatial_dim = int(np.sqrt(features.shape[-1]))
        features_torch = torch.Tensor(features).to("cuda:0")
        ss, ss_selector, vectors, values = self.eigen_decomposition(
            features_torch, head
        )
        # new_ss = ss.mean(dim=0, keepdim=True)
        # new_ss = new_ss.repeat(ss.shape[0], 1)
        new_ss = ss.reshape((-1, spatial_dim, spatial_dim))
        # new_ss = torch.roll(new_ss, spatial_dim//2, dims=2)
        # new_ss = new_ss - new_ss
        # new_ss[:, spatial_dim//2, spatial_dim//2] = torch.tensor(1.0).to(device)
        new_ss = TF.hflip(new_ss)
        new_ss = new_ss.reshape((-1, spatial_dim**2))
        if head < 0:
            final_features = self.eigen_reconstruction(
                new_ss, ss_selector, vectors, values
            )
        else:
            current_features[head : head + 1] = self.eigen_reconstruction(
                new_ss, ss_selector, vectors, values
            )
            final_features = current_features
        return final_features

    def fuse_ss(self, cur_features, previous_features, head=-1):
        spatial_dim = int(np.sqrt(cur_features.shape[-1]))
        features_torch = torch.Tensor(previous_features).to(device)
        # decompose prev features
        ss, ss_selector, vectors, values = self.eigen_decomposition(
            features_torch, head
        )
        # decompose current features
        _, ss_selector, vectors, values = self.eigen_decomposition(cur_features, head)
        # reconst. current features, with previous ss
        if head < 0:
            final_features = self.eigen_reconstruction(
                new_ss, ss_selector, vectors, values
            )
        else:
            cur_features[head : head + 1] = self.eigen_reconstruction(
                new_ss, ss_selector, vectors, values
            )
            final_features = cur_features
        return final_features

    def remove_attention(self, features):
        spatial_dim = int(np.sqrt(features.shape[-1]))
        features_torch = torch.Tensor(features).to(device)
        ss = self.get_ss(features_torch, head)
        return final_features

    def get_features(self, key, device, load_cross=False):
        if self.cur_timestep_in_range():
            location, is_cross = self.parse_layer_name(key)
            shortcut_key = self.key_to_shortcut(key)
            if shortcut_key in self.save_layer_names:
                if (load_cross and is_cross) or (not load_cross and not is_cross):
                    store_index = self.curr_timestep_index - self.first_timestep_index
                    features = self.attention_store[shortcut_key][store_index]
                    final_features = torch.Tensor(features).to(device)
                    # print("loading: {}, {}".format(key, features.shape))
                    return final_features, True
        return None, False

    def key_to_shortcut(self, key):
        split_key = key.split(".")
        if split_key[0] == "down_blocks":
            layer_type = "d"
        elif split_key[0] == "mid_block":
            layer_type = "m"
        elif split_key[0] == "up_blocks":
            layer_type = "u"
        return layer_type + split_key[1] + split_key[3]

    def parse_layer_name(self, name):
        if name.startswith("mid_block"):
            # hidden_size = model.unet.config.block_out_channels[-1]
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            # block_id = int(name[len("up_blocks.")])
            # hidden_size = list(reversed(model.unet.config.block_out_channels))[block_id]
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            # block_id = int(name[len("down_blocks.")])
            # hidden_size = model.unet.config.block_out_channels[block_id]
            place_in_unet = "down"
        else:
            raise ValueError("unknown layer type")
        is_cross = "attn2" in name
        return place_in_unet, is_cross

    def between_steps(self):
        """
        will be called after all attention layers were called
        """
        pass

    def get_store(self):
        return self.attention_store

    def aggregate_attention(
        self,
        res: int,
        from_where: List[str],
        # is_cross: bool,
        # select: int
    ) -> torch.Tensor:
        """Aggregates the attention across the different layers and heads at the specified resolution."""
        out = []
        attention_maps = self.get_store()
        num_pixels = res**2
        for location in from_where:
            for item in attention_maps[
                location
            ]:  # f"{location}_{'cross' if is_cross else 'self'}"
                if item.shape[2] == num_pixels:
                    # item is (b, num_pixels, seq) where for self_att seq = num_pixels
                    maps = item.reshape(
                        1, -1, num_pixels, num_pixels
                    )  # (1, b, num_pixels, num_pixels)
                    out.append(maps)
        out = torch.cat(out, dim=0)  # (layer, b, res**2, res**2)
        out = out.mean(dim=0)  # b, res**2, res**2
        return out

    def reset(self):
        self.cur_layer_step = 0
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = self.get_empty_store()
        self.freeze = False
        self.curr_timestep = 0
        self.curr_timestep_index = 0


class PLProcessor:
    def __init__(self, key, args, attention_store=None):
        self.key = key
        self.args = args
        self.symmetry_transform = False
        self.curr_timestamp = -1
        self.attention_store = attention_store

    def __call__(
        self,
        attn: attention_processor.Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query_tag = attn.head_to_batch_dim(query)
        key_tag = attn.head_to_batch_dim(key)
        value_tag = attn.head_to_batch_dim(value)

        valid = False
        if self.attention_store.inject_features:
            # attention_probs = self.get_attention_scores(query_tag, key_tag, attention_mask)
            features, valid = self.attention_store.get_features(
                self.key, value_tag.device
            )
        if valid:
            if self.args.inject_value == "query":
                query_tag = features
                attention_probs = self.get_attention_scores(
                    query_tag, key_tag, attention_mask
                )
            elif self.args.inject_value == "weight":
                attention_probs = features  # (2*b*h, seq, seq)
            else:
                raise NotImplementedError
        else:
            attention_probs = self.get_attention_scores(
                query_tag, key_tag, attention_mask
            )
        if self.attention_store is not None:
            if self.args.inject_value == "query":
                self.attention_store(query_tag, self.key, False)
            elif self.args.inject_value == "weight":
                self.attention_store(attention_probs, self.key, False)
            else:
                raise NotImplementedError
        hidden_states = torch.bmm(attention_probs, value_tag)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    def get_attention_scores(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dtype = query.dtype
        is_self_attention = query.shape[-2] == key.shape[-2]
        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0],
                query.shape[1],
                key.shape[1],
                dtype=query.dtype,
                device=query.device,
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1
        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=1 / sqrt(query.size(-1)),
        )
        del baddbmm_input
        if self.symmetry_transform and is_self_attention:  # Toeplitz decomposition
            alpha = 0.9
            attention_scores_sym = 0.5 * (
                attention_scores + attention_scores.transpose(-1, -2)
            )
            attention_scores_skew = 0.5 * (
                attention_scores - attention_scores.transpose(-1, -2)
            )
            sym_probs = attention_scores_sym.softmax(dim=-1)
            skew_probs = attention_scores_skew.softmax(dim=-1)
            attention_probs = alpha * (sym_probs) + (1 - alpha) * skew_probs
        else:
            attention_probs = attention_scores.softmax(dim=-1)

        del attention_scores
        attention_probs = attention_probs.to(dtype)
        return attention_probs


class PLPipeline(
    # StableDiffusionPipeline,
    DiffusionPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    StableDiffusionLoraLoaderMixin,
    IPAdapterMixin,
    FromSingleFileMixin,
):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        controlnet: ControlNetModel,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = False,
    ):
        super().__init__()

        if (
            hasattr(scheduler.config, "steps_offset")
            and scheduler.config.steps_offset != 1
        ):
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate(
                "steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if (
            hasattr(scheduler.config, "clip_sample")
            and scheduler.config.clip_sample is True
        ):
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate(
                "clip_sample not set", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        is_unet_version_less_0_9_0 = hasattr(
            unet.config, "_diffusers_version"
        ) and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse(
            "0.9.0.dev0"
        )
        is_unet_sample_size_less_64 = (
            hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        )
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate(
                "sample_size<64", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=False,
        )
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, StableDiffusionLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(
                prompt, padding="longest", return_tensors="pt"
            ).input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[
                -1
            ] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask
                )
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device),
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(
                    prompt_embeds
                )

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=prompt_embeds_dtype, device=device
            )

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

        if self.text_encoder is not None:
            if isinstance(self, StableDiffusionLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start

    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        image = self.control_image_processor.preprocess(
            image, height=height, width=width
        ).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma  # is 1.0 for ddim
        return latents

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        # control_guidance_start=0.0,
        # control_guidance_end=1.0,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if callback_steps is not None and (
            not isinstance(callback_steps, int) or callback_steps <= 0
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs
            for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (
            not isinstance(prompt, str) and not isinstance(prompt, list)
        ):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        # if not isinstance(control_guidance_start, (tuple, list)):
        #     control_guidance_start = [control_guidance_start]

        # if not isinstance(control_guidance_end, (tuple, list)):
        #     control_guidance_end = [control_guidance_end]

        if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
            raise ValueError(
                "Provide either `ip_adapter_image` or `ip_adapter_image_embeds`. Cannot leave both `ip_adapter_image` and `ip_adapter_image_embeds` defined."
            )

        if ip_adapter_image_embeds is not None:
            if not isinstance(ip_adapter_image_embeds, list):
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be of type `list` but is {type(ip_adapter_image_embeds)}"
                )
            elif ip_adapter_image_embeds[0].ndim not in [3, 4]:
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be a list of 3D or 4D tensors but is {ip_adapter_image_embeds[0].ndim}D"
                )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 1.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        lora_guidance_scale=None,
        image=None,
        strength=1.0,
        lora_target=None,
        debug_lora=False,
        lora_timestamp=999,
        lora_output_timestep=None,
        lora_timestamp_cutoff=None,
        lora_iter_per_timestep=1,
        noise_label=False,
        inject_features=False,
        attention_store=None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        control_image=None,
        zs=None,  # ddpm inversion parameters
        norm_gradients=True,
        return_intermeds_timesteps=None,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)
        lora_output = []
        intermediate_latents = []
        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        use_control_net = self.controlnet is not None and control_image is not None
        if use_control_net:
            controlnet = (
                self.controlnet._orig_mod
                if is_compiled_module(self.controlnet)
                else self.controlnet
            )
            control_guidance_start, control_guidance_end = (
                [control_guidance_start],
                [control_guidance_end],
            )
        # to deal with lora scaling and other possible forward hooks
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            # control_guidance_start,
            # control_guidance_end,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        if use_control_net:
            global_pool_conditions = (
                controlnet.config.global_pool_conditions
                if isinstance(controlnet, ControlNetModel)
                else controlnet.nets[0].config.global_pool_conditions
            )
            guess_mode = guess_mode or global_pool_conditions
        if lora_output_timestep is not None:
            lora_output_timestep = np.array(lora_output_timestep)
        if return_intermeds_timesteps is not None:
            return_intermeds_timesteps = np.array(return_intermeds_timesteps)
        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None)
            if self.cross_attention_kwargs is not None
            else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )
        # 4. Prepare controlnet Image
        if use_control_net:
            control_image = self.prepare_image(
                image=control_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
            # height, width = image.shape[-2:]
        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )
        # 5. Prepare latent variables
        if image is not None:
            timesteps, num_inference_steps = self.get_timesteps(
                num_inference_steps, strength, device
            )  # calc step given strength
            latent_timestep = timesteps[:1].repeat(
                batch_size * num_images_per_prompt
            )  # fetch first timestep
            init_latents = self.vae.encode(image).latent_dist.sample(
                generator
            )  # encode image
            init_latents = self.vae.config.scaling_factor * init_latents
            noise = randn_tensor(
                init_latents.shape,
                generator=generator,
                device=device,
                dtype=prompt_embeds.dtype,
            )
            init_latents = self.scheduler.add_noise(
                init_latents, noise, latent_timestep
            )  # noise it to first timestep
            latents = init_latents
            orig_noise = noise.detach().clone()
        else:
            num_channels_latents = self.unet.config.in_channels
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )
            orig_noise = latents.detach().clone()
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(
                batch_size * num_images_per_prompt
            )
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)
        # debug to see what lora outputs without the full denoising loop
        if debug_lora:
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2)
                if self.do_classifier_free_guidance
                else latents
            )
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, lora_timestamp
            )
            # predict using lora
            predictions = self.unet(
                latent_model_input,
                lora_timestamp,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=self.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            image = self.vae.decode(
                predictions / self.vae.config.scaling_factor,
                return_dict=False,
                generator=generator,
            )[0]
            # (b, 3, 512 512)
            return image
        # 7.2 Create tensor stating which controlnets to keep
        if use_control_net:
            controlnet_keep = []
            for i in range(len(timesteps)):
                keeps = [
                    1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                    for s, e in zip(control_guidance_start, control_guidance_end)
                ]
                controlnet_keep.append(
                    keeps[0] if isinstance(controlnet, ControlNetModel) else keeps
                )
        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        # is_unet_compiled = is_compiled_module(self.unet)
        # is_controlnet_compiled = is_compiled_module(self.controlnet)
        # is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")
        ddpmi = zs is not None
        if ddpmi:
            op = timesteps[-zs.shape[0] :]
            t_to_idx = {int(v): k for k, v in enumerate(timesteps[-zs.shape[0] :])}
            actual_n_steps = len(op)
            actual_timesteps = op
        else:
            actual_n_steps = num_inference_steps
            actual_timesteps = timesteps
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=actual_n_steps) as progress_bar:
            for i, t in enumerate(actual_timesteps):
                if attention_store is not None:
                    attention_store.set_timesstep(i / (len(actual_timesteps) - 1), i)
                if self.interrupt:
                    continue
                ########## START: lora guidance
                if lora_timestamp_cutoff is not None:
                    do_lora = (
                        lora_timestamp_cutoff[0]
                        <= i / (len(actual_timesteps) - 1)
                        <= lora_timestamp_cutoff[1]
                    )
                else:
                    do_lora = True
                adapters = self.get_active_adapters()
                if len(adapters) == 0:
                    do_lora = False
                if (lora_guidance_scale is not None) and do_lora:
                    with torch.enable_grad():
                        if self.do_classifier_free_guidance:
                            my_prompt_embeds = prompt_embeds[
                                len(prompt_embeds) // 2 :, ...
                            ]  # only use the conditional embeddings
                        else:
                            my_prompt_embeds = prompt_embeds
                        for _ in range(lora_iter_per_timestep):
                            for j, adapter in enumerate(adapters):
                                latents = latents.clone().detach().requires_grad_(True)
                                weights = [0.0 for _ in adapters]
                                weights[j] = 1.0
                                self.set_adapters(adapters, adapter_weights=weights)
                                # compute loss
                                output = self.unet(
                                    latents,
                                    t,
                                    encoder_hidden_states=my_prompt_embeds,
                                    timestep_cond=timestep_cond,
                                    cross_attention_kwargs=self.cross_attention_kwargs,
                                    # down_block_additional_residuals=down_block_res_samples,
                                    # mid_block_additional_residual=mid_block_res_sample,
                                    added_cond_kwargs=added_cond_kwargs,
                                    return_dict=False,
                                )[0]
                                if lora_output_timestep is not None:
                                    indices = actual_timesteps[lora_output_timestep]
                                    if torch.isin(indices, t).any():
                                        lora_output.append(output)
                                    # if t == actual_timesteps[lora_output_timestep]:
                                    #     lora_output.append(output)
                                target = lora_target[
                                    j
                                ]  # target should already be encoded and of the right dimensions
                                ### noisy label
                                if noise_label and j == 0:
                                    # noise = randn_tensor(target.shape, generator=generator, device=device, dtype=target.dtype)
                                    target = self.scheduler.add_noise(
                                        target, orig_noise, t
                                    )
                                ###
                                loss = F.mse_loss(output, target)
                                # print("{:02d}:{}".format(j, loss.item()))
                                # get gradient
                                cond_grad = torch.autograd.grad(loss, latents)[0]
                                # normalize it
                                if norm_gradients:
                                    denom = (
                                        torch.sqrt(
                                            torch.sum(cond_grad**2, dim=(1, 2, 3))
                                        )
                                        + 1e-5
                                    )
                                    cond_grad = cond_grad / denom[:, None, None, None]
                                # modify the latents based on this gradient
                                latents = (
                                    latents.detach() - cond_grad * lora_guidance_scale
                                )
                        self.set_adapters(
                            adapters, adapter_weights=[0.0] * len(adapters)
                        )
                        ########## START: CFG (with lora guidance)
                        if attention_store is not None:
                            attention_store.inject_features = inject_features
                        # expand the latents if we are doing classifier free guidance
                        latent_model_input = (
                            torch.cat([latents] * 2)
                            if self.do_classifier_free_guidance
                            else latents
                        )
                        latent_model_input = self.scheduler.scale_model_input(
                            latent_model_input, t
                        )

                        # controlnet(s) inference
                        if use_control_net:
                            if guess_mode and self.do_classifier_free_guidance:
                                # Infer ControlNet only for the conditional batch.
                                control_model_input = latents
                                control_model_input = self.scheduler.scale_model_input(
                                    control_model_input, t
                                )
                                controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                            else:
                                control_model_input = latent_model_input
                                controlnet_prompt_embeds = prompt_embeds
                            if isinstance(controlnet_keep[i], list):
                                cond_scale = [
                                    c * s
                                    for c, s in zip(
                                        controlnet_conditioning_scale,
                                        controlnet_keep[i],
                                    )
                                ]
                            else:
                                controlnet_cond_scale = controlnet_conditioning_scale
                                if isinstance(controlnet_cond_scale, list):
                                    controlnet_cond_scale = controlnet_cond_scale[0]
                                cond_scale = controlnet_cond_scale * controlnet_keep[i]

                            down_block_res_samples, mid_block_res_sample = (
                                self.controlnet(
                                    control_model_input,
                                    t,
                                    encoder_hidden_states=controlnet_prompt_embeds,
                                    controlnet_cond=control_image,
                                    conditioning_scale=cond_scale,
                                    guess_mode=guess_mode,
                                    return_dict=False,
                                )
                            )

                            if guess_mode and self.do_classifier_free_guidance:
                                # Inferred ControlNet only for the conditional batch.
                                # To apply the output of ControlNet to both the unconditional and conditional batches,
                                # add 0 to the unconditional batch to keep it unchanged.
                                down_block_res_samples = [
                                    torch.cat([torch.zeros_like(d), d])
                                    for d in down_block_res_samples
                                ]
                                mid_block_res_sample = torch.cat(
                                    [
                                        torch.zeros_like(mid_block_res_sample),
                                        mid_block_res_sample,
                                    ]
                                )
                        else:
                            down_block_res_samples = None
                            mid_block_res_sample = None

                        # predict the noise residual
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            timestep_cond=timestep_cond,
                            cross_attention_kwargs=self.cross_attention_kwargs,
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]
                        # perform guidance
                        if self.do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + self.guidance_scale * (
                                noise_pred_text - noise_pred_uncond
                            )

                        if (
                            self.do_classifier_free_guidance
                            and self.guidance_rescale > 0.0
                        ):
                            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                            noise_pred = rescale_noise_cfg(
                                noise_pred,
                                noise_pred_text,
                                guidance_rescale=self.guidance_rescale,
                            )

                        torch.cuda.empty_cache()
                        if attention_store is not None:
                            attention_store.inject_features = False
                        ########## END: CFG (with lora guidance)
                ########## END: lora guidance
                ########## START: original diffusion prediciton (+CFG)
                else:
                    if attention_store is not None:
                        attention_store.inject_features = inject_features
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        torch.cat([latents] * 2)
                        if self.do_classifier_free_guidance
                        else latents
                    )
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )

                    # controlnet(s) inference
                    if use_control_net:
                        if guess_mode and self.do_classifier_free_guidance:
                            # Infer ControlNet only for the conditional batch.
                            control_model_input = latents
                            control_model_input = self.scheduler.scale_model_input(
                                control_model_input, t
                            )
                            controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                        else:
                            control_model_input = latent_model_input
                            controlnet_prompt_embeds = prompt_embeds

                        if isinstance(controlnet_keep[i], list):
                            cond_scale = [
                                c * s
                                for c, s in zip(
                                    controlnet_conditioning_scale, controlnet_keep[i]
                                )
                            ]
                        else:
                            controlnet_cond_scale = controlnet_conditioning_scale
                            if isinstance(controlnet_cond_scale, list):
                                controlnet_cond_scale = controlnet_cond_scale[0]
                            cond_scale = controlnet_cond_scale * controlnet_keep[i]

                        down_block_res_samples, mid_block_res_sample = self.controlnet(
                            control_model_input,
                            t,
                            encoder_hidden_states=controlnet_prompt_embeds,
                            controlnet_cond=control_image,
                            conditioning_scale=cond_scale,
                            guess_mode=guess_mode,
                            return_dict=False,
                        )

                        if guess_mode and self.do_classifier_free_guidance:
                            # Inferred ControlNet only for the conditional batch.
                            # To apply the output of ControlNet to both the unconditional and conditional batches,
                            # add 0 to the unconditional batch to keep it unchanged.
                            down_block_res_samples = [
                                torch.cat([torch.zeros_like(d), d])
                                for d in down_block_res_samples
                            ]
                            mid_block_res_sample = torch.cat(
                                [
                                    torch.zeros_like(mid_block_res_sample),
                                    mid_block_res_sample,
                                ]
                            )
                    else:
                        down_block_res_samples = None
                        mid_block_res_sample = None

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )

                    if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(
                            noise_pred,
                            noise_pred_text,
                            guidance_rescale=self.guidance_rescale,
                        )
                    if attention_store is not None:
                        attention_store.inject_features = False
                    ###### START: lora readout
                    if lora_output_timestep is not None:
                        indices = actual_timesteps[lora_output_timestep]
                        if torch.isin(indices, t).any():
                            adapters = self.get_active_adapters()
                            if len(adapters) > 0:
                                for j, adapter in enumerate(adapters):
                                    weights = [0.0 for _ in adapters]
                                    weights[j] = 1.0
                                    self.set_adapters(adapters, adapter_weights=weights)
                                    if self.do_classifier_free_guidance:
                                        my_prompt_embeds = prompt_embeds[
                                            len(prompt_embeds) // 2 :, ...
                                        ]  # only use the positive embeddings
                                    else:
                                        my_prompt_embeds = prompt_embeds
                                    result = self.unet(
                                        latents,
                                        t,
                                        encoder_hidden_states=my_prompt_embeds,
                                        timestep_cond=timestep_cond,
                                        cross_attention_kwargs=self.cross_attention_kwargs,
                                        added_cond_kwargs=added_cond_kwargs,
                                        return_dict=False,
                                    )[0]
                                    lora_output.append(result)
                                self.set_adapters(
                                    adapters, adapter_weights=[0.0] * len(adapters)
                                )
                    ###### END: lora readout
                ########## END: original diffusion prediciton (+CFG)
                # compute the previous noisy sample x_t -> x_t-1
                
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(actual_timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
                if return_intermeds_timesteps is not None:
                    indices = actual_timesteps[return_intermeds_timesteps]
                    if torch.isin(indices, t).any():
                        intermediate_latents.append(latents.detach())
        if use_control_net:
            # If we do sequential model offloading, let's offload unet and controlnet
            # manually for max memory savings
            if (
                hasattr(self, "final_offload_hook")
                and self.final_offload_hook is not None
            ):
                self.unet.to("cpu")
                self.controlnet.to("cpu")
                torch.cuda.empty_cache()

        if not output_type == "latent":
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor,
                return_dict=False,
                generator=generator,
            )[0]
            do_denormalize = [True] * image.shape[0]
            if len(lora_output) > 0:
                lora_output_decoded = self.vae.decode(
                    torch.cat(lora_output) / self.vae.config.scaling_factor,
                    return_dict=False,
                    generator=generator,
                )[0]
                do_denormalize_lora = [True] * lora_output_decoded.shape[0]
            if len(intermediate_latents) > 0:
                intermediate_decoded = self.vae.decode(
                    torch.cat(intermediate_latents) / self.vae.config.scaling_factor,
                    return_dict=False,
                    generator=generator,
                )[0]
                do_denormalize_intermediates = [True] * intermediate_decoded.shape[0]
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            do_denormalize = [False] * image.shape[0]
            if len(lora_output) > 0:
                lora_output_decoded = torch.cat(lora_output)
                do_denormalize_lora = [False] * lora_output_decoded.shape[0]
            if len(intermediate_latents) > 0:
                intermediate_decoded = torch.cat(intermediate_latents)
                do_denormalize_intermediates = [False] * intermediate_decoded.shape[0]
            # has_nsfw_concept = None

        # if has_nsfw_concept is None:
        #     do_denormalize = [True] * image.shape[0]
        # else:
        #     do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            if len(lora_output) > 0:
                lora_output_decoded = self.image_processor.postprocess(
                    lora_output_decoded,
                    output_type=output_type,
                    do_denormalize=do_denormalize_lora,
                )
                if len(intermediate_latents) > 0:
                    intermediate_decoded = self.image_processor.postprocess(
                        intermediate_decoded,
                        output_type=output_type,
                        do_denormalize=do_denormalize_intermediates,
                    )
                    return (image, lora_output_decoded, intermediate_decoded)
                else:
                    return (image, lora_output_decoded, None)
            else:
                if len(intermediate_latents) > 0:
                    intermediate_decoded = self.image_processor.postprocess(
                        intermediate_decoded,
                        output_type=output_type,
                        do_denormalize=do_denormalize_intermediates,
                    )
                    return (image, None, intermediate_decoded)
                else:
                    return (image, None, None)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)
