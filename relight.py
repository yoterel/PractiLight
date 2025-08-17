import argparse
import yaml
import numpy as np
import torch
import shutil
from pathlib import Path
from PIL import Image
import cv2
from diffusers import (
    DDIMScheduler,
    ControlNetModel,
)
from torchvision import transforms
from intrinsic.pipeline import load_models as intrinsic_load_models
from intrinsic.pipeline import run_pipeline as intrinsic_run_pipeline
from pipeline_practilight import PLPipeline, register_attention_control, AttentionStore
import gsoup
from utils import color_transfer, image_to_depth

def parse_command_line():
    """
    parses command line option, preferrably everything is in a YAML file (config)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument("--prompt", type=str, help="prompt")
    parser.add_argument("--n_prompt", type=str, help="negative prompt")
    parser.add_argument("--n_steps", type=int, help="number of sampling steps")
    parser.add_argument("--cfg", type=float, help="classifier free guidance scale")
    parser.add_argument("--seed", type=int, help="seed")
    parser.add_argument("--checkpoint", type=str, help="checkpoint location")
    parser.add_argument("--gamma", type=float, help="Override learning rate")
    parser.add_argument("--gamma_t", type=float, help="Override learning rate")
    parser.add_argument(
        "--controlnet_guidance_scale", type=float, help="Override learning rate"
    )
    parser.add_argument("--controlnet_t", type=float, help="Override learning rate")
    parser.add_argument("--inject_t", type=float, help="Override learning rate")
    parser.add_argument("--mode", type=int, help="Override learning rate")
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="folder to store downloaded models (will be created if doesnt exist)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="folder to output results (will be created if doesnt exist)",
    )
    parser.add_argument("--device", type=str, help="device to use (cuda:0, cpu, etc)")
    args = parser.parse_args()
    return args

class Struct:
    """
    used to convert a dictionary to struct
    """
    def __init__(self, **entries):
        self.__dict__.update(entries)

def override_config(config, args):
    """
    allow overriding some configurations with command line options
    """
    if args.prompt is not None:
        config["prompt"] = args.prompt
    if args.n_prompt is not None:
        config["n_prompt"] = args.n_prompt
    if args.n_steps is not None:
        config["n_steps"] = args.n_steps
    if args.cfg is not None:
        config["cfg"] = args.cfg
    if args.seed is not None:
        config["seed"] = args.seed
    if args.n_steps is not None:
        config["n_steps"] = args.n_steps
    if args.checkpoint is not None:
        config["checkpoint"] = args.checkpoint
    if args.gamma is not None:
        config["gamma"] = args.gamma
    if args.gamma_t is not None:
        config["gamma_t"] = args.gamma_t
    if args.controlnet_guidance_scale is not None:
        config["controlnet_guidance_scale"] = args.controlnet_guidance_scale
    if args.controlnet_t is not None:
        config["controlnet_t"] = args.controlnet_t
    if args.inject_t is not None:
        config["inject_t"] = args.inject_t
    if args.mode is not None:
        config["mode"] = args.mode
    if args.cache_dir is not None:
        config["cache_dir"] = args.cache_dir
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.device is not None:
        config["device"] = args.device
    return config


def main():
    # parse CLI arguments / YAML config
    pargs = parse_command_line()
    with open(pargs.config, "r") as f:
        config = yaml.safe_load(f)
    
    # override YAML config with CLI args if provided
    args = override_config(config, pargs)
    args = Struct(**config)
    torch.hub.set_dir(str(args.cache_dir))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # load models
    if args.mixed_precision:
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=weight_dtype,
        cache_dir=args.cache_dir,
    )
    albedo_model = intrinsic_load_models("v2")
    pipeline = PLPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=weight_dtype,
        controlnet=controlnet,
        cache_dir=args.cache_dir,
    )
    pipeline.scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        cache_dir=args.cache_dir,
    )
    pipeline.enable_model_cpu_offload()
    # pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=args.device)

    # fetch latest lora weights
    lora_names = []
    for i, lora_exp_checkpoint in enumerate(args.folder_to_load_lora_checkpoint):
        dirs = [
            d for d in sorted(Path(lora_exp_checkpoint).glob("checkpoint*"))
        ]
        dirs = sorted(dirs, key=lambda x: int(x.stem.split("-")[1]))
        lora_weights_path = None

        if len(dirs) > 0:
            lora_weights_path = Path(
                dirs[args.lora_checkpoint_index[i]],
                "pytorch_lora_weights.safetensors",
            )
        else:
            raise FileNotFoundError
        lora_name = "lora_{:02d}".format(i)
        lora_names.append(lora_name)
        pipeline.load_lora_weights(lora_weights_path, lora_name)
    adapters = pipeline.get_active_adapters()
    if len(adapters) != 0:
        pipeline.set_adapters(lora_names, adapter_weights=[0.0] * len(lora_names))

    # set seed
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    torch.manual_seed(args.seed)
    adapters = pipeline.get_active_adapters()

    # set custom processor to save attention
    attention_store = AttentionStore(args)
    register_attention_control(
        pipeline, args, default=False, attention_store=attention_store
    )
    # generate image to relight from seed
    attention_store.reset()
    gen_state = generator.get_state()
    orig_images, lora_readouts, _ = pipeline(
        prompt=[args.prompt] * args.batch_size,
        negative_prompt=[args.prompt_negative] * args.batch_size,
        num_inference_steps=args.num_steps,
        generator=generator,
        guidance_scale=args.cfg_scale,
        control_image=None,
        output_type="np",
        return_dict=False,
        lora_output_timestep=-1,
        attention_store=attention_store,
    )
    lora_readout = gsoup.to_8b(lora_readouts[0])
    orig_image = gsoup.to_8b(orig_images[0])
    gsoup.save_image(orig_image, Path(output_dir, "original.png"))
    gsoup.save_image(lora_readout, Path(output_dir, "original_readout.png"))
    control_signal_valid = False
    if args.control_signal is not None:
        if Path(args.control_signal).exists():
           control_signal_valid = True

    if control_signal_valid:
        # preprocess control signal
        image_transforms = transforms.Compose(
            [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(512),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        control_signal_image = gsoup.load_image(Path(args.control_signal))
        if control_signal_image.shape[-1] == 1:
            control_signal_image = np.tile(control_signal_image, (1, 1, 3))
        control_signal_image = gsoup.color_to_gray(control_signal_image, keep_channels=True)
        control_signal_pil = Image.fromarray(control_signal_image)
        lora_target = image_transforms(control_signal_pil)
        lora_target = lora_target.repeat(args.batch_size, 1, 1, 1)
        lora_target_embeds = pipeline.vae.encode(lora_target.to(dtype=weight_dtype, device=args.device)).latent_dist.sample()
        lora_target_embeds = lora_target_embeds * pipeline.vae.config.scaling_factor
        lora_target_embeds = [lora_target_embeds]
    else:
        print("Control signal not provided, generating depth map for proxy scene.")
        depth, depth_model = image_to_depth(orig_image[None, ...], args, None)
        disparity = 1 / (depth[0] + 1e-5)
        disparity_normalized = gsoup.map_to_01(disparity)
        gsoup.save_image(disparity_normalized, Path(output_dir, "proxy.png"))
        # gsoup.save_image(orig_image, Path(output_dir, "orig.png")
        print("saved at: {}. open booth.blend with Blender to create a control signal.".format(Path(output_dir, "proxy.png")))
        return
    # prepare controlnet edge map
    # prepare albedo map
    results = intrinsic_run_pipeline(albedo_model, gsoup.to_float(orig_image))
    albedo = results["hr_alb"]
    albedo = np.clip(albedo, 0, 1)
    albedo = gsoup.to_8b(albedo)
    albedo = Image.fromarray(albedo).resize((512, 512))
    image_for_canny = np.array(albedo)
    # prepare albedo edge map
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(image_for_canny, low_threshold, high_threshold)
    image = image[:, :, None]
    control_image = np.concatenate([image, image, image], axis=2)
    control_image = Image.fromarray(control_image)
    # prepare control edge map
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(control_signal_image, low_threshold, high_threshold)
    image = image[:, :, None].astype(bool)
    # merge edge maps
    control_image_orig = np.array(control_image)[:, :, 0:1].astype(bool)
    control_image_new = control_image_orig | image
    control_image_new = control_image_new.astype(np.uint8) * 255
    control_image_np = np.concatenate(
        [control_image_new, control_image_new, control_image_new], axis=2
    )
    gsoup.save_image(control_image_np, Path(output_dir, "controlnet_input.png"))
    control_image_pil = Image.fromarray(control_image_np)
    # relight
    attention_store.freeze = True
    generator.set_state(gen_state)
    result, final_lora_readout, intermeds = pipeline(
        prompt=[args.prompt] * args.batch_size,
        negative_prompt=[args.prompt_negative] * args.batch_size,
        num_inference_steps=args.num_steps,
        generator=generator,
        guidance_scale=args.cfg_scale,
        latents=None,
        lora_guidance_scale=args.lora_guidance_scale,
        lora_target=lora_target_embeds,
        output_type="np",
        control_image=control_image_pil,
        control_guidance_start=args.controlnet_t[0],
        control_guidance_end=args.controlnet_t[1],
        controlnet_conditioning_scale=args.controlnet_guidance_scale,
        lora_output_timestep=args.lora_output_timesteps,
        lora_timestamp_cutoff=args.lora_timestamp_cutoff,
        lora_iter_per_timestep=args.lora_iter_per_timestep,
        inject_features=True,
        attention_store=attention_store,
        norm_gradients=args.norm_gradients,
        return_intermeds_timesteps=args.return_intermeds_timesteps,
        return_dict=False,
    )
    result = gsoup.to_8b(result[0])
    gsoup.save_image(result, Path(output_dir, "relit_raw.png"))
    final_lora_readouts = []
    intermediates = []
    if final_lora_readout is not None:
        for readout in final_lora_readout:
            final_lora_readouts.append(gsoup.to_8b(readout))
    if intermeds is not None:
        for intermed in intermeds:
            intermediates.append(gsoup.to_8b(intermed))
    # post process color correction
    color_corrected = color_transfer(orig_image, result, clip=True, preserve_paper=False)
    # save results
    gsoup.save_image(color_corrected, Path(output_dir, "relit.png"))


if __name__ == "__main__":
    main()
