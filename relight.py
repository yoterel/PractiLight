import argparse
import yaml
import numpy as np
import torch
import shutil
from pipline_practilight import PLPipeline, register_attention_control, AttentionStore


def parse_command_line():
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


def override_config(config, args):
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
    # 1. Parse CLI arguments
    args = parse_command_line()

    # 2. Load YAML config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # 3. Override YAML config with CLI args if provided
    args = override_config(config, args)
    torch.hub.set_dir(str(args.cache_dir))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # 4. load models
    if args.mixed_precision:
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float16,
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
            d for d in sorted(Path("output", lora_exp_checkpoint).glob("checkpoint*"))
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

    # 5. set seed
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    torch.manual_seed(args.seed)
    adapters = pipeline.get_active_adapters()

    # 6. set custom processor to save attention
    attention_store = AttentionStore(args)
    register_attention_control(
        pipeline, args, default=False, attention_store=attention_store
    )

    attention_store.reset()
    gen_state = generator.get_state()
    no_lora_images, lora_readout, _ = pipeline(
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

    no_lora_images = gsoup.to_8b(no_lora_images)
    # albedo
    results = intrinsic_run_pipeline(albedo_model, gsoup.to_float(no_lora_images[0]))
    albedo = results["hr_alb"]
    albedo = np.clip(albedo, 0, 1)
    albedo = gsoup.to_8b(albedo)
    albedo = Image.fromarray(albedo).resize((512, 512))
    image_for_canny = np.array(albedo)
    # edge map
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(image_for_canny, low_threshold, high_threshold)
    image = image[:, :, None]
    control_image = np.concatenate([image, image, image], axis=2)

    control_image = Image.fromarray(control_image)
    # control image 2
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
    control_image_new = control_image_new.astype(np.uint8) * 255
    control_image_np = np.concatenate(
        [control_image_new, control_image_new, control_image_new], axis=2
    )
    control_image_pil = Image.fromarray(control_image_np)
    control_images.append(control_image_np)
    # 10. relight
    attention_store.freeze = True
    generator.set_state(gen_state)
    result, final_lora_readout, intermeds = pipeline(
        prompt=[args.prompt] * args.batch_size,
        negative_prompt=[args.prompt_negative] * args.batch_size,
        num_inference_steps=args.num_steps,
        generator=generator,
        guidance_scale=args.cfg_scale,
        latents=inverted_latents,
        lora_guidance_scale=args.lora_guidance_scale,
        lora_target=labels_embeds,
        output_type="np",
        control_image=control_image_pil,
        control_guidance_start=args.controlnet_timestamp_cutoff[0],
        control_guidance_end=args.controlnet_timestamp_cutoff[1],
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
    # 11. post process color correction
    cc = []
    if len(lora_images) == len(guidance_images):  # sanity check for correcting color
        for iii in range(len(lora_images)):
            cur_guide = guidance_images[iii][0]
            cur_relit = lora_images[iii][0]
            orig = no_lora_images[0]  # assumes batch size 1
            r_equals_b = np.all(cur_guide[..., 0] == cur_guide[..., 1])
            b_equals_g = np.all(cur_guide[..., 1] == cur_guide[..., 2])
            if r_equals_b and b_equals_g:
                color_corrected = color_transfer(
                    orig, cur_relit, clip=True, preserve_paper=False
                )
            else:
                color_corrected = color_transfer(
                    cur_guide, cur_relit, clip=True, preserve_paper=False
                )
            cc.append(color_corrected)
    # 12. save results


if __name__ == "__main__":
    main()
