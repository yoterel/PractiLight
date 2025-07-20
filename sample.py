import argparse
import logging
import math
import os
import os.path as osp
import random
import shutil
from pathlib import Path
import wandb
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from torch.utils.data import Dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from controlnet_aux import HEDdetector
from datasets import SynthSceneDataset, BasicDataset, SynthMultiLightDataset, MIIW_Dataset, DiffusionDB_Dataset, MixerDataset, ExistingDataset, LightStageMultiObjectDatasetMultiLight
from pipeline_ltd import LTDPipeline, register_attention_control, AttentionStore
import diffusers
from diffusers import (
    AutoencoderKL, 
    DDPMScheduler, 
    DiffusionPipeline, 
    UNet2DConditionModel, 
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    DDIMInverseScheduler,
    ControlNetModel,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.utils import load_image
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
import copy
import cv2
from PIL import Image
from PIL.ImageOps import exif_transpose
import json
import random
from ddpm_inversion import inversion_forward_process
# from torchvision.transforms.functional import pil_to_tensor
# from torchvision.transforms.functional import to_pil_image
# from torchvision.transforms.functional import resize
# from torchvision.transforms.functional import center_crop
from pdb import set_trace
from safetensors.torch import load_file
import yaml
import gsoup
from myutils import unet_pred_to_image, compute_permutations, create_nice_grid_image, color_transfer
from intrinsic.pipeline import load_models as intrinsic_load_models
from intrinsic.pipeline import run_pipeline as intrinsic_run_pipeline


def image_to_depth(images, args, model):
    if model is None:
        model_zoe_n = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True)
        model = model_zoe_n.to(args.device)
    # print(torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384"))  # Triggers fresh download of MiDaS repo
    # print(torch.hub.help("isl-org/ZoeDepth", "ZoeD_N"))
    # midas = torch.hub.load("intel-isl/MiDaS", "DPT_BEiT_L_384")
    my_tensor = gsoup.to_torch(gsoup.to_float(images), device=args.device).permute(0, 3, 1, 2)
    depth = model.infer(my_tensor)  # (b, 1, 512, 512)
    depth_numpy = gsoup.to_np(depth[:, 0, :, :, None])
    depth_numpy = np.tile(depth_numpy, (1, 1, 1, 3))
    return depth_numpy, model


def inversion(pipeline, generator, init_images, args, weight_dtype, attention_store, reconstruct=True):
    # encode image
    latents = pipeline.vae.encode(init_images.to(dtype=weight_dtype, device=args.device)).latent_dist.sample()
    latents = latents * pipeline.vae.config.scaling_factor
    # forward pass
    if attention_store is not None:
        attention_store.freeze = True  # quick hack for ddpm_inversion not uinsg our pipeline directly
    print("inverting images")
    if args.inference_inversion_type == "ddim_inversion":
        pipeline.scheduler = DDIMInverseScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler", cache_dir=args.cache_dir
        )
        # invert    
        inv_latents, _, _ = pipeline(
            prompt=[args.inference_prompt_inv]*args.inference_batch_size,
            negative_prompt=[args.inference_prompt_negative_inv]*args.inference_batch_size,
            num_inference_steps=args.inference_num_steps,
            generator=generator,
            guidance_scale=args.cfg_scale_inv,
            output_type="latent",
            return_dict=False,
            latents=latents,
        )
        pipeline.scheduler = DDIMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler", cache_dir=args.cache_dir
        )
    elif args.inference_inversion_type == "ddpm_inversion":
        pipeline.scheduler.set_timesteps(args.inference_num_steps)
        _, zs, wts = inversion_forward_process(pipeline, latents, etas=1.0, 
                                                prompt=args.inference_prompt_inv, 
                                                cfg_scale=args.cfg_scale_inv, 
                                                prog_bar=True, device=args.device,
                                                num_inference_steps=args.inference_num_steps)
    else:
        raise NotImplementedError
    # set back to DDIM scheduler
    # backward pass (reconstruct)
    support_zs = None
    if attention_store is not None:
        attention_store.freeze = False  # quick hack for ddpm_inversion not uinsg our pipeline directly
    if (reconstruct):
        print("reconstructing...")
        if args.inference_inversion_type == "ddim_inversion":
            reconstruction, lora_readout, _ = pipeline(
                prompt=[args.inference_prompt]*args.inference_batch_size,
                negative_prompt=[args.inference_prompt_negative]*args.inference_batch_size,
                num_inference_steps=args.inference_num_steps,
                generator=generator,
                guidance_scale=args.cfg_scale,
                output_type="np",
                return_dict=False,
                lora_output_timestep=-1,
                attention_store=attention_store,
                latents=inv_latents,
            )
        elif args.inference_inversion_type == "ddpm_inversion":
            print("skipping {:03d} steps...".format(args.inference_ddpm_inversion_skip))
            sel_t = args.inference_num_steps-args.inference_ddpm_inversion_skip
            inv_latents = wts[sel_t:sel_t+1]
            support_zs = zs[:sel_t]
            reconstruction, lora_readout, _ = pipeline(
                prompt=[args.inference_prompt]*args.inference_batch_size,
                negative_prompt=[args.inference_prompt_negative]*args.inference_batch_size,
                num_inference_steps=args.inference_num_steps,
                generator=generator,
                guidance_scale=args.cfg_scale,
                output_type="np",
                return_dict=False,
                lora_output_timestep=-1,
                attention_store=attention_store,
                latents=inv_latents,
                zs=support_zs,
            )
            # w0, _ = inversion_reverse_process(pipeline, 
            #                                     xT=wts[sel_t:sel_t+1], 
            #                                     zs=zs[:sel_t],
            #                                     prompts=[args.inference_prompt], 
            #                                     cfg_scales=[args.cfg_scale],
            #                                     device=args.device)
            # reconstruction = pipeline.vae.decode(w0 / pipeline.vae.config.scaling_factor, return_dict=False)[0]
            # reconstruction = reconstruction.clamp(min=-1.0,max=1.0).float()
            # reconstruction = gsoup.to_numpy(reconstruction.permute(0,2,3,1))
            # reconstruction = gsoup.map_range(reconstruction, -1.0, 1.0, 0.0, 1.0)
            # lora_readout = reconstruction
        else:
            raise NotImplementedError
    else:
        reconstruction = None
        lora_readout = None
    return inv_latents, reconstruction, lora_readout, support_zs

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
    
def main():
    # parse cmdline / config
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    pargs = parser.parse_args()
    config = yaml.safe_load(open(pargs.config))
    args = Struct(**config)
    exp_name = args.default_exp_name
    job_index = args.default_job_index
    if "SLURM_JOB_ID" in os.environ:
        print("slurm job id={}".format(os.environ["SLURM_JOB_ID"]))
        exp_name = os.environ["SLURM_JOB_ID"]
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        print("slurm job index={}".format(os.environ["SLURM_ARRAY_TASK_ID"]))
        job_index = os.environ["SLURM_ARRAY_TASK_ID"]
    job_index = int(job_index)
    cache_path = Path("/CT/ObjectInjection/work/light_transport_diffuse/models")
    torch.hub.set_dir(str(cache_path))
    if args.inference_hyper_parameter_grid_search:  # do a sweep
        if job_index < args.inference_grid_search_start_index:  # unless this job index is smaller than starting point
            return
        perms, names, shortcut, myshape = compute_permutations(config, args.permutation_file, args.shortcuts_file)  # compute all permutations
        new_config = copy.deepcopy(config)
        shortcut_list = []
        for value, name in zip(perms[job_index], names):  # create a new config with this specific permutattion
            new_config[name] = value
            if name in shortcut:  # create a shortcut name if possible
                shortcut_list.append(shortcut[name])
        multi_index = np.unravel_index(job_index, myshape, order='C')
        shortcut_name = ""
        if len(shortcut_list) == len(multi_index):  # dont do this if there are permutation variables without shortcuts
            for i in range(len(shortcut_list)):
                shortcut_name +=shortcut_list[i]+"-"+"{:02d}".format(multi_index[i])+"_"
        output_dir = Path("output", "inference", args.inference_parameter_grid_search_name, "{:03d}_".format(job_index)+shortcut_name)
        output_dir.mkdir(exist_ok=True, parents=True)
        config_dst = Path(output_dir, "config.yml")
        with open(config_dst, 'w') as f:  # dump new config in the experiment folder
            yaml.dump(new_config, f)
        print("grid search permutation index: {} / {}".format(job_index, len(perms)))
        config = yaml.safe_load(open(config_dst))  # parse it (though we couldve directly used the one we saved)
        args = Struct(**config)
    elif args.inference_parallel_jobs:
        output_dir = Path("output", "inference", args.inference_output_dir_name)
        output_dir.mkdir(exist_ok=True, parents=True)
        config_dst = Path(output_dir, "config.yml")
        shutil.copy(pargs.config, config_dst)
    else:
        output_dir = Path("output", "inference", exp_name)
        output_dir.mkdir(exist_ok=True, parents=True)
        config_dst = Path(output_dir, "config.yml")
        shutil.copy(pargs.config, config_dst)
    if args.mixed_precision:
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32
    # define pipeline
    if args.inference_use_controlnet:
        if args.inference_controlnet_type == "canny":
            loc ="sd-controlnet-canny"
        elif args.inference_controlnet_type == "scribble":
            loc = "sd-controlnet-scribble"
        elif args.inference_controlnet_type == "hed":
            loc = "sd-controlnet-hed"
        else:
            raise NotImplementedError
        controlnet = ControlNetModel.from_pretrained("lllyasviel/{}".format(loc),
                                                    # torch_dtype=torch.float16,
                                                    # use_safetensors=True,
                                                    cache_dir=args.cache_dir)
        
        albedo_model = intrinsic_load_models('v2')
    else:
        controlnet = None
    depth_model = None
    pipeline = LTDPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
            controlnet=controlnet,
            cache_dir=args.cache_dir
            # use_safetensors=True
            # safety_checker=None,
        )
    pipeline.scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", cache_dir=args.cache_dir
    )
    if args.inference_use_controlnet:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(args.device)
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=args.device)
    # fetch latest lora weights
    lora_names = []
    for i, lora_exp_checkpoint in enumerate(args.inference_exp_to_load_lora_checkpoint):
        dirs = [d for d in sorted(Path("output", lora_exp_checkpoint).glob("checkpoint*"))]
        dirs = sorted(dirs, key=lambda x: int(x.stem.split("-")[1]))
        lora_weights_path = None
        
        if len(dirs) > 0:
            lora_weights_path = Path(dirs[args.inference_lora_checkpoint_index[i]], "pytorch_lora_weights.safetensors")
        else:
            raise FileNotFoundError
        lora_name = "lora_{:02d}".format(i)
        lora_names.append(lora_name)
        pipeline.load_lora_weights(lora_weights_path, lora_name)
    if len(lora_names) != 0:
        pipeline.set_adapters(lora_names, adapter_weights=[0.0]*len(lora_names))
    
    # init datasets
    if type(args.seed) == list:
        generator = torch.Generator(device=args.device).manual_seed(args.seed[0])
        torch.manual_seed(args.seed[0])
    else:
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        torch.manual_seed(args.seed)
    image_dataloader, guidance_dataloader, prompt_dataloader, guidance_dataset = init_datasets(args)
    
    # start guided inference
    adapters = pipeline.get_active_adapters()
    if len(adapters) != 0:
        pipeline.set_adapters(adapters, adapter_weights=[0.0]*len(adapters))
    pipeline.set_progress_bar_config(disable=False)
    if args.inference_store_attention:
        attention_store = AttentionStore(args) 
        register_attention_control(pipeline, args, default=False, attention_store=attention_store)
    else:
        attention_store = None
        register_attention_control(pipeline, args, default=True)
    # exclude existing output
    if not args.inference_force_create:
        do_rename = False
        if do_rename:
            output_types = ["control", "guidance", "inverted", "readouts_gen", "readouts_orig", "relit"]
            for output_type in output_types:
                existing = [x for x in Path(output_dir, output_type).glob("*.png")]
                for x in existing:
                    parts = x.stem.split("_")
                    parts[0] = "{:04d}".format(int(parts[0]))
                    new_stem = "_".join(parts)
                    x.rename(Path(x.parent, new_stem + x.suffix))
        existing_files = [x for x in sorted(Path(output_dir, "inverted").glob("*.png"))]
        if prompt_dataloader is not None:
            existing_mask = np.zeros(len(prompt_dataloader), dtype=bool)
            all_numbers = np.arange(len(prompt_dataloader))
        elif type(args.seed) == list:
            existing_mask = np.zeros(len(args.seed), dtype=bool)
            all_numbers = np.arange(len(args.seed))
        else:
            existing_mask = np.zeros(len(existing_files))
            all_numbers = np.arange(len(existing_files))
        if len(existing_files) > 0:
            existing_numbers = np.array([int(x.stem.split("_")[0]) for x in existing_files])
            existing_mask[existing_numbers] = True
    # loop over prompts/seeds
    DEBUG = False
    if prompt_dataloader is None:
        if type(args.seed) == list:
            n_seeds = len(args.seed)
        else:
            n_seeds = args.inference_num_seeds
    else:
        if args.inference_num_seeds < 0:
            n_seeds = len(prompt_dataloader)  
        else:
            n_seeds = args.inference_num_seeds + args.inference_skip_seeds
        prompt_data_loader_iter = iter(prompt_dataloader)
    for i in range(n_seeds):
        print("seed: {} / {}".format(i+1, n_seeds))
        if prompt_dataloader is not None:
            prompt_batch = next(prompt_data_loader_iter)
            args.seed = int(prompt_batch["seed"][0])
            # args.inference_num_steps = args.inference_num_steps #prompt_batch["step"][0]
            args.cfg_scale = prompt_batch["cfg"][0]
            args.inference_prompt = prompt_batch["prompt"][0]
            generator = torch.Generator(device=args.device).manual_seed(args.seed)
        elif type(args.seed) == list:
            generator = torch.Generator(device=args.device).manual_seed(args.seed[i])
        if args.inference_adaptive_guidance:
            lora_scale = args.inference_lora_guidance_scale * (100 / args.inference_num_steps)
        else:
            if args.inference_num_steps <= 10:
                lora_scale = 5.0
            else:
                lora_scale = args.inference_lora_guidance_scale
        if args.inference_parallel_jobs and i//args.inference_max_seeds_per_job != job_index:
            print("skipping, not my job")
            continue
        if not args.inference_force_create:
            if existing_mask[i]:
                print("skipping, exists")
                continue
        if i < args.inference_skip_seeds:
            print("skipping, according to inference_skip_seeds")
            continue
        # compute total images to generate
        if image_dataloader is None:
            n_images_to_generate = args.inference_max_images
        else:
            n_images_to_generate = len(image_dataloader) if args.inference_max_images < 0 else args.inference_max_images
            image_data_loader_iter = iter(image_dataloader)
        for ii in range(n_images_to_generate):
            gen_state = generator.get_state()
            guidance_images = []
            hint_images = []
            lora_images = []
            gt_images = []
            control_images = []
            gt_targets = []
            final_lora_readouts = []
            intermediates = []
            generator.set_state(gen_state)
            # generate with no lora guidance
            print("generating image with no guidance")
            if args.inference_store_attention:
                if not DEBUG or ii==0:
                    attention_store.reset()
            if args.inference_from_image:
                # todo: support batch size > 1
                image_batch = next(image_data_loader_iter)
                gt_images.append(unet_pred_to_image(image_batch["images"], False, False))
                if "images2" in image_batch.keys():
                    gt_targets.append(unet_pred_to_image(image_batch["images2"], False, False))
                images = torch.Tensor(image_batch["images"]).to(dtype=weight_dtype, device=args.device)
                if args.inference_inversion:
                    inverted_latents, reconstruction, lora_readout, support_zs = inversion(pipeline, generator, images, args, weight_dtype, attention_store)
                    no_lora_images = gsoup.to_8b(reconstruction)
                    sd_edit_images = None
                else:
                    sd_edit_images = images
                    inverted_latents = None
                    support_zs = None
                    no_lora_images, lora_readout, _ = pipeline(
                                        prompt=[args.inference_prompt]*args.inference_batch_size,
                                        negative_prompt=[args.inference_prompt_negative]*args.inference_batch_size,
                                        num_inference_steps=args.inference_num_steps, 
                                        generator=generator,
                                        guidance_scale=args.cfg_scale,
                                        image=sd_edit_images,
                                        latents=inverted_latents,
                                        strength=args.inference_sdedit_strength,
                                        output_type="np",
                                        return_dict=False,
                                        lora_output_timestep=-1,
                                        attention_store=attention_store,
                                        )
                    no_lora_images = gsoup.to_8b(no_lora_images)
            else:
                sd_edit_images = None
                inverted_latents = None
                support_zs = None
                no_lora_images, lora_readout, _ = pipeline(
                                        prompt=[args.inference_prompt]*args.inference_batch_size,
                                        negative_prompt=[args.inference_prompt_negative]*args.inference_batch_size,
                                        num_inference_steps=args.inference_num_steps, 
                                        generator=generator,
                                        guidance_scale=args.cfg_scale,
                                        control_image=None,
                                        output_type="np",
                                        return_dict=False,
                                        lora_output_timestep=-1,
                                        attention_store=attention_store,
                                        )
                no_lora_images = gsoup.to_8b(no_lora_images)
            if args.inference_use_controlnet:
                if args.inference_controlnet_use_albedo:
                    results = intrinsic_run_pipeline(albedo_model, gsoup.to_float(no_lora_images[0]))
                    albedo = results['hr_alb']
                    albedo = np.clip(albedo, 0, 1)
                    albedo = gsoup.to_8b(albedo)
                    albedo = Image.fromarray(albedo).resize((512, 512))
                    image_for_canny = np.array(albedo)
                else:
                    image_for_canny = no_lora_images[0]
                if args.inference_controlnet_type == "canny":
                    low_threshold = 100
                    high_threshold = 200
                    image = cv2.Canny(image_for_canny, low_threshold, high_threshold)
                    image = image[:, :, None]
                    control_image = np.concatenate([image, image, image], axis=2)
                    # control_images.append(control_image)
                    control_image = Image.fromarray(control_image)
                # elif args.inference_controlnet_type == "scribble":
                #     hed = HEDdetector.from_pretrained("lllyasviel/Annotators", cache_dir=args.cache_dir)
                #     control_image = hed(no_lora_images[0], scribble=True)
                # elif args.inference_controlnet_type == "hed":
                #     hed = HEDdetector.from_pretrained("lllyasviel/Annotators", cache_dir=args.cache_dir)
                #     control_image = hed(no_lora_images[0])
                else:
                    raise NotImplementedError
            else:
                control_image_pil = None
            if args.inference_output_type == "attmaps":
                assert args.inference_store_attention
                print("computing attention maps")
                # result = attention_store.pca_store()
                # attention_store.do_pca_viz(result, output_dir)
                # attention_store.viz_steady_state(no_lora_images, Path(output_dir, "attmaps", "ss"), i ,ii, grid=True, threshold=True)
                # attention_store.viz_edit(no_lora_images, Path(output_dir, "attmaps", "ss_edit"), i ,ii)
                # attention_store.viz_col_sum(no_lora_images, Path(output_dir, "attmaps", "col_sum"), i ,ii)
                attention_store.visualize_per_layer(no_lora_images, Path(output_dir, "attmaps_{:02d}_{:02d}".format(i, ii)))
                # continue
            if args.inference_guidance_dataset_type == "manual":
                if guidance_dataloader is None:
                    if args.inference_from_image:
                        my_images = np.vstack(gt_images)
                    else:
                        my_images = no_lora_images
                    depth, depth_model = image_to_depth(my_images, args, depth_model)
                    for iii in range(len(depth)):
                        disparity = 1 / (depth[iii] + 1e-5)
                        # disparity_clipped = disparity.clip(0, 1)
                        disparity_normalized = gsoup.map_to_01(disparity)
                        # disparity_normalized_clipped = disparity_normalized.clip(0, 1)
                        # normalized_depth = gsoup.map_to_01(depth[i])
                        # normalized_depth_inverted = 1.0 - normalized_depth
                        # gsoup.write_exr(depth[i], Path(output_dir, "depth_{:02d}.exr".format(i)))
                        # gsoup.write_exr(disparity, Path(output_dir, "disp_{:02d}.exr".format(i)))
                        # gsoup.save_image(normalized_depth, Path(output_dir, "depth_norm_{:02d}.png".format(i)))
                        # gsoup.save_image(normalized_depth_inverted, Path(output_dir, "depth_norm_inv_{:02d}.png".format(i)))
                        # gsoup.save_image(disparity_clipped, Path(output_dir, "disp_clip_{:02d}.png".format(i)))
                        gsoup.save_image(disparity_normalized, Path(output_dir, "disp_norm_{:04d}_{:04d}_{:04d}.png".format(i, ii, iii)))
                        # gsoup.save_image(disparity_normalized_clipped, Path(output_dir, "disp_norm_clip_{:02d}.png".format(i)))
                        gsoup.save_image(my_images[iii], Path(output_dir, "orig_{:04d}_{:04d}_{:04d}.png".format(i, ii, ii)))
                    continue
            assert len(adapters) <= 2
            if len(adapters) == 2:
                _, lora_albedo = np.split(lora_readout, len(adapters))
            if len(adapters) > 0:
                lora_readout = gsoup.to_8b(lora_readout)
            # don't store anymore attention activations
            if args.inference_store_attention:
                attention_store.freeze = True
            # loop over guidance signals
            if len(adapters) != 0 or args.force_create_guidance:
                if guidance_dataloader is None:
                    n_guidance_signals = 1
                else:
                    n_guidance_signals = len(guidance_dataloader) if args.inference_num_guidance < 0 else args.inference_num_guidance
                    guidance_dataloader_iter = iter(guidance_dataloader)
            else:
                n_guidance_signals = 1
            for iii in range(n_guidance_signals):
                print("generating image with guidance: {} / {}".format(iii+1, n_guidance_signals))
                generator.set_state(gen_state)
                if len(adapters) != 0 or args.force_create_guidance:
                    if guidance_dataloader is None:
                        cur_label = image_batch["labels2"]
                    else:
                        if args.inference_guidance_dataset_type == "mixer":
                            guidance_dataset.set_image(no_lora_images[0])
                        elif args.inference_guidance_dataset_type == "existing":
                            guidance_dataset.set_index(i)
                        guidance_batch = next(guidance_dataloader_iter)
                        cur_label = guidance_batch["labels"]
                    guidance_image = unet_pred_to_image(cur_label, False, False)
                    if "hints" in guidance_batch.keys():
                        cur_hint_images = guidance_batch["hints"]
                    # create control image
                    if args.inference_use_controlnet:
                        if args.inference_controlnet_use_light:
                            low_threshold = 100
                            high_threshold = 200
                            image = cv2.Canny(guidance_image[0], low_threshold, high_threshold)
                            image = image[:, :, None].astype(bool)
                            combine = True
                            if combine:
                                control_image_orig = np.array(control_image)[:, :, 0:1].astype(bool)
                                control_image_new = control_image_orig | image
                            else:
                                control_image_new = image
                            control_image_new = control_image_new.astype(np.uint8)*255
                            control_image_np = np.concatenate([control_image_new, control_image_new, control_image_new], axis=2)
                        else:
                            control_image_np = np.array(control_image)
                        control_image_pil = Image.fromarray(control_image_np)
                        control_images.append(control_image_np)  
                    guidance_images.append(guidance_image)
                    if "hints" in guidance_batch.keys():
                        hint_images.append(cur_hint_images)
                    if len(adapters) != 0:
                        primary_labels = cur_label.repeat(args.inference_batch_size, 1, 1, 1)
                        primary_labels_embeds = pipeline.vae.encode(primary_labels.to(dtype=weight_dtype, device=args.device)).latent_dist.sample()
                        primary_labels_embeds = primary_labels_embeds * pipeline.vae.config.scaling_factor
                        labels_embeds = [primary_labels_embeds]
                    ### add second adapter readout to labels
                        if len(adapters) > 1:
                            lora_albedo = torch.Tensor(lora_albedo).to(dtype=weight_dtype, device=args.device)
                            albedo = lora_albedo.permute(0, 3, 1, 2)
                            normalize = transforms.Normalize([0.5], [0.5])
                            albedo = normalize(albedo)
                            label_albedo = pipeline.vae.encode(albedo.to(dtype=weight_dtype, device=args.device)).latent_dist.sample()
                            label_albedo = label_albedo * pipeline.vae.config.scaling_factor
                            labels_embeds.append(label_albedo)
                    else:
                        labels_embeds = None
                else:
                    labels_embeds=None
                result, final_lora_readout, intermeds = pipeline(prompt=[args.inference_prompt]*args.inference_batch_size,
                                                    negative_prompt=[args.inference_prompt_negative]*args.inference_batch_size,
                                                    num_inference_steps=args.inference_num_steps, 
                                                    generator=generator,
                                                    eta=args.inference_eta,
                                                    guidance_scale=args.cfg_scale,
                                                    image=sd_edit_images,
                                                    latents=inverted_latents,
                                                    strength=args.inference_sdedit_strength,  # 1.0: maximum noise
                                                    lora_guidance_scale=lora_scale,
                                                    lora_target=labels_embeds,
                                                    output_type="np",
                                                    control_image=control_image_pil,
                                                    control_guidance_start=args.inference_controlnet_timestamp_cutoff[0],
                                                    control_guidance_end=args.inference_controlnet_timestamp_cutoff[1],
                                                    controlnet_conditioning_scale=args.inference_controlnet_guidance_scale,
                                                    lora_output_timestep=args.lora_output_timesteps,
                                                    lora_timestamp_cutoff=args.inference_lora_timestamp_cutoff,
                                                    lora_iter_per_timestep=args.inference_lora_iter_per_timestep,
                                                    identity_guidance_mode=args.inference_identity_guidance_mode,
                                                    noise_label=args.noise_label,
                                                    inject_features=args.inference_inject_features,
                                                    attention_store=attention_store,
                                                    zs=support_zs,
                                                    norm_gradients=args.norm_gradients,
                                                    return_intermeds_timesteps=args.return_intermeds_timesteps,
                                                    return_dict=False)
                lora_images.append(gsoup.to_8b(result))
                if final_lora_readout is not None:
                    for readout in final_lora_readout:
                        final_lora_readouts.append(gsoup.to_8b(readout))
                if intermeds is not None:
                    for intermed in intermeds:
                        intermediates.append(gsoup.to_8b(intermed))
            # post process correct color
            cc = []
            if len(lora_images) == len(guidance_images):  # sanity check for correcting color
                for iii in range(len(lora_images)):
                    cur_guide = guidance_images[iii][0]
                    cur_relit = lora_images[iii][0]
                    orig = no_lora_images[0]  # assumes batch size 1
                    r_equals_b = np.all(cur_guide[..., 0] == cur_guide[..., 1])
                    b_equals_g = np.all(cur_guide[..., 1] == cur_guide[..., 2])
                    if r_equals_b and b_equals_g:
                        color_corrected = color_transfer(orig, cur_relit, clip=True, preserve_paper=False)
                    else:
                        color_corrected = color_transfer(cur_guide, cur_relit, clip=True, preserve_paper=False)
                    cc.append(color_corrected)
                #####
            if args.inference_output_type == "grid":
                assert len(adapters) > 0  # will only work if we actually did some guidance...
                image_grid = create_nice_grid_image(gt_images, guidance_images, lora_images, no_lora_images, lora_readout, len(adapters), args)
                gsoup.save_image(image_grid, Path(output_dir, "result_{:04d}_{:04d}.png".format(i, ii)))
            elif args.inference_output_type == "seperate" or args.inference_output_type == "attmaps":
                if args.inference_from_image:
                    stacked = np.vstack(gt_images)
                    for iii in range(len(stacked)):
                        gsoup.save_image(stacked[iii], Path(output_dir, "input", "{:04d}_{:04d}_{:04d}".format(i, ii, iii)))
                if len(gt_targets) > 0:
                    stacked = np.vstack(gt_targets)
                    for iii in range(len(stacked)):
                        gsoup.save_image(stacked[iii], Path(output_dir, "target", "{:04d}_{:04d}_{:04d}".format(i, ii, iii)))
                if len(guidance_images) > 0:
                    stacked = np.vstack(guidance_images)
                    for iii in range(len(stacked)):
                        gsoup.save_image(stacked[iii], Path(output_dir, "guidance", "{:04d}_{:04d}_{:04d}".format(i, ii, iii)))
                if len(hint_images) > 0:
                    stacked = np.vstack(hint_images)[0]
                    for iii in range(len(stacked)):
                        gsoup.save_image(stacked[iii], Path(output_dir, "hints", "{:04d}_{:04d}_{:04d}".format(i, ii, iii)))
                if len(lora_images) > 0:
                    stacked = np.vstack(lora_images)
                    for iii in range(len(stacked)):
                        gsoup.save_image(stacked[iii], Path(output_dir, "relit", "{:04d}_{:04d}_{:04d}".format(i, ii, iii)))
                if len(final_lora_readouts) > 0:
                    for iii in range(len(final_lora_readouts)):
                        gsoup.save_image(final_lora_readouts[iii], Path(output_dir, "readouts_gen", "{:04d}_{:04d}_{:04d}".format(i, ii, iii)))
                if len(intermediates) > 0:
                    for iii in range(len(intermediates)):
                        gsoup.save_image(intermediates[iii], Path(output_dir, "intermediates", "{:04d}_{:04d}_{:04d}".format(i, ii, iii)))
                if len(no_lora_images) > 0:
                    for iii in range(len(no_lora_images)):
                        gsoup.save_image(no_lora_images[iii], Path(output_dir, "inverted", "{:04d}_{:04d}_{:04d}".format(i, ii, iii)))
                if lora_readout is not None:
                    if len(lora_readout) > 0:
                        for iii in range(len(lora_readout)):
                            gsoup.save_image(lora_readout[iii], Path(output_dir, "readouts_orig", "{:04d}_{:04d}_{:04d}".format(i, ii, iii)))
                if len(cc) > 0:
                    for iii in range(len(cc)):
                        gsoup.save_image(cc[iii], Path(output_dir, "relit_cc", "{:04d}_{:04d}_{:04d}".format(i, ii, iii)))
                
                if len(control_images) > 0:
                    for iii in range(len(control_images)):
                        gsoup.save_image(control_images[iii], Path(output_dir, "control", "{:04d}_{:04d}_{:04d}.png".format(i ,ii, iii)))

def init_datasets(args):
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="tokenizer", 
    )
    if args.inference_guidance_dataset_type == "synth_scene":
        guidance_dataset = SynthSceneDataset(
            data_root = args.inference_guidance_dataset,
            is_train = False,
            tokenizer = tokenizer,
            num_train_imgs=None,
            prompt = "",
            direct_light=True,
            clean_data=True,
            random_color=args.inference_random_guidance_color
        )
        guidance_dataloader = torch.utils.data.DataLoader(
            guidance_dataset,
            shuffle=False,
            collate_fn=lambda examples: SynthSceneDataset.collate_fn(examples),
            batch_size=1,
            num_workers=0,
        )
    elif args.inference_guidance_dataset_type == "synth_multilight":
        guidance_dataset = SynthMultiLightDataset(
            data_root = args.inference_guidance_dataset,
            is_train = False,
            tokenizer = tokenizer,
            # num_train_imgs=None,
            prompt = "",
            direct_light=True,
            clean_data=True,
            filter_data=args.inference_best_inversions_file
        )
        guidance_dataloader = torch.utils.data.DataLoader(
            guidance_dataset,
            shuffle=False,
            collate_fn=lambda examples: SynthMultiLightDataset.collate_fn(examples),
            batch_size=1,
            num_workers=0,
        )
    elif args.inference_guidance_dataset_type == "test_patterns":
        guidance_dataset = BasicDataset(
            data_root = args.inference_guidance_dataset,
            random_color=args.inference_random_guidance_color,
        )
        guidance_dataloader = torch.utils.data.DataLoader(
            guidance_dataset,
            shuffle=False,
            collate_fn=lambda examples: BasicDataset.collate_fn(examples),
            batch_size=1,
            num_workers=0,
        )
    elif args.inference_guidance_dataset_type == "manual":
        guidance_dataloader = None
        guidance_dataset = None
        if args.inference_guidance_dataset is not None:
            guidance_folder = Path(args.inference_guidance_dataset)
            if guidance_folder.exists():
                # guidance_files = [x for x in guidance_folder.glob("*")]
                # if len(guidance_files) == len(no_lora_images):
                guidance_dataset = BasicDataset(data_root=guidance_folder,
                                                random_color=args.inference_random_guidance_color,
                                                to_gray=True)
                guidance_dataloader = torch.utils.data.DataLoader(
                    guidance_dataset,
                    shuffle=False,
                    collate_fn=lambda examples: BasicDataset.collate_fn(examples),
                    batch_size=1,
                    num_workers=0,
                )
    elif args.inference_guidance_dataset_type == "mixer":
        guidance_dataset = MixerDataset(modes=args.inference_guidance_dataset_mode, 
                                        random_color=args.inference_random_guidance_color)
        guidance_dataloader = torch.utils.data.DataLoader(
                guidance_dataset,
                shuffle=True,
                collate_fn=lambda examples: MixerDataset.collate_fn(examples),
                batch_size=1,
                num_workers=0,
            )
    elif args.inference_guidance_dataset_type == "existing":
        guidance_dataset = ExistingDataset(args.inference_guidance_dataset,
                                            guidance_per_image=args.inference_num_guidance,
                                            single_channel=True,
                                            stride=args.inference_guidance_stride,
                                            start_index=args.inference_guidance_start_index)
        guidance_dataloader = torch.utils.data.DataLoader(
                guidance_dataset,
                shuffle=False,
                collate_fn=lambda examples: ExistingDataset.collate_fn(examples),
                batch_size=1,
                num_workers=0,
            )
    else:
        guidance_dataloader = None
        guidance_dataset = None
    if args.inference_image_dataset_type == "synth_scene":
        image_dataset = SynthSceneDataset(
            data_root = args.inference_image_dataset,
            is_train = False,
            tokenizer = tokenizer,
            num_train_imgs=None,
            prompt = "",
            direct_light=True,
            clean_data=True
        )
        image_dataloader = torch.utils.data.DataLoader(
            image_dataset,
            shuffle=False,
            collate_fn=lambda examples: SynthSceneDataset.collate_fn(examples),
            batch_size=1,
            num_workers=0,
        )
    elif args.inference_image_dataset_type == "synth_multilight":
        image_dataset = SynthMultiLightDataset(
            data_root = args.inference_image_dataset,
            is_train = False,
            tokenizer = tokenizer,
            # num_train_imgs=None,
            prompt = "",
            direct_light=True,
            clean_data=True,
            filter_data=args.inference_best_inversions_file
        )
        image_dataloader = torch.utils.data.DataLoader(
            image_dataset,
            shuffle=False,
            collate_fn=lambda examples: SynthMultiLightDataset.collate_fn(examples),
            batch_size=1,
            num_workers=0,
        )
    #######
    elif args.inference_image_dataset_type == "real_multilight":
        image_dataset = LightStageMultiObjectDatasetMultiLight(
            data_root = args.inference_image_dataset,
            is_train = False,
            # tokenizer = tokenizer,
            num_train_imgs=None,
            # prompt = "",
            preprocessed_path = args.inference_image_dataset_alt,
            seed = args.seed,
            hdr_pipeline=True,
            n_light_sources=2,
            random_exposure=False,
            random_color=True,
        )
        image_dataloader = torch.utils.data.DataLoader(
            image_dataset,
            shuffle=True,
            collate_fn=lambda examples: LightStageMultiObjectDatasetMultiLight.collate_fn(examples),
            batch_size=1,
            num_workers=0,
        )
    #######
    elif args.inference_image_dataset_type == "image_collection":
        my_folder = Path(args.inference_image_dataset)
        if my_folder.exists():
            # guidance_files = [x for x in guidance_folder.glob("*")]
            # if len(guidance_files) == len(no_lora_images):
            image_dataset = BasicDataset(data_root=my_folder)
            image_dataloader = torch.utils.data.DataLoader(
                image_dataset,
                shuffle=False,
                collate_fn=lambda examples: BasicDataset.collate_fn(examples),
                batch_size=1,
                num_workers=0,
            )
        else:
            image_dataloader = None
    elif args.inference_image_dataset_type == "MIIW_dataset":
        my_folder1 = Path(args.inference_image_dataset)
        my_folder2 = Path(args.inference_image_dataset_alt)
        image_dataset = MIIW_Dataset(my_folder1, my_folder2)
        image_dataloader = torch.utils.data.DataLoader(
                image_dataset,
                shuffle=False,
                collate_fn=lambda examples: MIIW_Dataset.collate_fn(examples),
                batch_size=1,
                num_workers=0,
            )
    else:
        image_dataloader = None
    if args.inference_prompt_dataset_type == "DiffusionDB":
        prompt_dataset = DiffusionDB_Dataset(args.inference_prompt_dataset, filter_file=args.inference_prompt_filter)
        prompt_dataloader = torch.utils.data.DataLoader(
            prompt_dataset,
            shuffle=False,
            collate_fn=lambda examples: DiffusionDB_Dataset.collate_fn(examples),
            batch_size=1,
            num_workers=0,
        )
    else:
        prompt_dataloader = None
    return image_dataloader, guidance_dataloader, prompt_dataloader, guidance_dataset

if __name__=="__main__":
    main()

# sanity: generate with no guidance, same geenrator, and decode outside of pipeline (should yield same as above)
# result = pipeline([my_prompt]*args.inference_batch_size,
#                     num_inference_steps=50, 
#                     generator=generator,
#                     guidance_scale=7.5,
#                     output_type="latent")
# with torch.no_grad():
#     images = pipeline.vae.decode(result.images / pipeline.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
#     # image_grid = unet_pred_to_image(images)
#     # gsoup.save_image(image_grid, Path(output_dir, "vanilla2.png"))
#     do_denormalize = [True] * images.shape[0]
#     images = pipeline.image_processor.postprocess(images, output_type="np", do_denormalize=do_denormalize)
#     image_grid = gsoup.image_grid(images, 1, args.inference_batch_size)
#     gsoup.save_image(image_grid, Path(output_dir, "vanilla2.png"))