import argparse
import copy
import logging
import os
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from collections import OrderedDict
import lpips
import torch_fidelity

import torch
import torch.utils.checkpoint
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import DistributedDataParallelKwargs 

from models.sit import SiT_models
from models.autoencoder import vae_models
from utils import load_encoders, denormalize_latents

from samplers import euler_sampler, euler_maruyama_sampler
from dataset import CustomH5Dataset
import wandb
import torch.distributed as dist
from torchvision.utils import make_grid
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize

from PIL import Image

torch_home = os.environ.setdefault("TORCH_HOME", os.path.expanduser("~/.cache/torch"))
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
torch.hub.set_dir(os.path.join(torch_home, "hub"))

logger = get_logger(__name__)

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

def preprocess_raw_image(x, enc_type):
    resolution = x.shape[-1]
    if 'clip' in enc_type:
        x = x / 255.
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    elif 'dinov1' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')

    return x


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z - latents_bias) * latents_scale # normalize
    return z 


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

    # NOTE: Let's also perform EMA on BN buffers, if later during inference we want to use original model stats
    #       We can also directly load that stats.
    ema_buffers = OrderedDict(ema_model.named_buffers())
    model_buffers = OrderedDict(model.named_buffers())

    for name, buffer in model_buffers.items():
        name = name.replace("module.", "")
        if buffer.dtype in (torch.float32, torch.float64):  # Apply EMA only to float buffers
            ema_buffers[name].mul_(decay).add_(buffer.data, alpha=1 - decay)
        else:
            ema_buffers[name].copy_(buffer)  # Direct copy for non-float buffers


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):    
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    # accelerator = Accelerator(
    #     gradient_accumulation_steps=args.gradient_accumulation_steps,
    #     mixed_precision=args.mixed_precision,
    #     log_with=args.report_to,
    #     project_config=accelerator_project_config,
    # )
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs]
    )

    # set up the logger and checkpoint dirs
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Create model:
    if args.vae == "f8d4":
        assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
        latent_size = args.resolution // 8
        in_channels = 4
    elif args.vae == "f16d32":
        assert args.resolution % 16 == 0, "Image size must be divisible by 16 (for the VAE encoder)."
        latent_size = args.resolution // 16
        in_channels = 32
    else:
        raise NotImplementedError()

    if args.enc_type != None:
        encoders, encoder_types, architectures = load_encoders(
            args.enc_type, device, args.resolution
        )
    else:
        raise NotImplementedError()
    z_dims = [encoder.embed_dim for encoder in encoders] if args.enc_type != 'None' else [0]

    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT_models[args.model](
        input_size=latent_size,
        in_channels=in_channels,
        num_classes=args.num_classes,
        class_dropout_prob=args.cfg_prob,
        z_dims=z_dims,
        encoder_depth=args.encoder_depth,
        **block_kwargs
    )

    # make a copy of the model for EMA
    model = model.to(device)
    ema = copy.deepcopy(model).to(device)  # Create an EMA of the model for use after training

    if args.ldm_only:
        requires_grad(model, True)
        if hasattr(model, "pixel_final_layer"):
            requires_grad(model.pixel_final_layer, False)
        if hasattr(model, "pil_embedder"):
            requires_grad(model.pil_embedder, False)
        if hasattr(model, "pixel_fusion_gamma"):
            model.pixel_fusion_gamma.requires_grad = False
        if accelerator.is_main_process:
            logger.info("Training mode: ldm-only first stage. Pixel branch is disabled.")
    else:
        requires_grad(model, False)
        if hasattr(model, 'pixel_final_layer'):
            requires_grad(model.pixel_final_layer, True)
            if accelerator.is_main_process:
                logger.info("Training mode: pixel branch. Successfully unfrozen pixel_final_layer.")
        else:
            logger.error("Error: model has no attribute 'pixel_final_layer'. Please check your model definition.")
        if hasattr(model, 'pixel_fusion_gamma'):
            model.pixel_fusion_gamma.requires_grad = True
        

    # Load the VAE latents-stats, also squeeze all singleton dimensions
    latents_stats = torch.load(args.vae_ckpt.replace(".pt", "-latents-stats.pt"))
    latents_scale = latents_stats["latents_scale"].squeeze().to(device)
    latents_bias = latents_stats["latents_bias"].squeeze().to(device)
    model.init_bn(latents_bias=latents_bias, latents_scale=latents_scale)

    # Load the vae model for inference
    vae = vae_models[args.vae]().to(device)
    vae_ckpt = torch.load(args.vae_ckpt, map_location=device)
    vae.load_state_dict(vae_ckpt)
    del vae_ckpt

    # Apply SyncBN if more than 1 GPU is used, although the stats are not changed during training
    if accelerator.use_distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"Trainable SiT Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # define the optimizers for SiT and VAE separately
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Setup data:
    train_dataset = CustomH5Dataset(data_dir=args.data_dir, vae_latents_name=args.vae_latents_name)
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights

    model.eval()
    ema.eval()

    # resume:
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt_path = f'{args.continue_train_exp_dir}/checkpoints/{ckpt_name}'
        # If the checkpoint exists, we load the checkpoint and resume the training
        ckpt = torch.load(
            ckpt_path,
            map_location='cpu',
        )
        # model.load_state_dict(ckpt['model'])
        state_dict = ckpt['model']
        if args.ignore_pixel_final_layer:
            state_dict = {
                k: v for k, v in state_dict.items()
                if not k.startswith("pixel_final_layer.")
            }
        msg = model.load_state_dict(state_dict, strict=False) 
        print("Missing keys:", msg.missing_keys)
        ema_state_dict = ckpt['ema']
        if args.ignore_pixel_final_layer:
            ema_state_dict = {
                k: v for k, v in ema_state_dict.items()
                if not k.startswith("pixel_final_layer.")
            }
        msg = ema.load_state_dict(ema_state_dict, strict=False)
        print("Missing keys EMA:", msg.missing_keys)
        # optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']

    # Allow larger cache size for DYNAMo compilation
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 512

    # Model compilation for better performance
    if args.compile:
        model = torch.compile(model, backend="inductor", mode="default")

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name="gradient-pass-through",
            config=tracker_config,
            init_kwargs={
                "wandb": {"name": f"{args.exp_name}"}
            },
        )

    log_interval = int(args.log_interval)
    running_sit_loss = 0.0
    running_denoising_loss = 0.0
    running_proj_loss = 0.0
    running_jit_loss = 0.0
    running_lpips_loss_vgg = 0.0
    running_grad_norm = 0.0
    running_steps = 0

    # Labels to condition the model with (feel free to change):
    sample_batch_size = 64 // accelerator.num_processes
    ys = torch.randint(1000, size=(sample_batch_size,), device=device)
    ys = ys.to(device)
    # Create sampling noise:
    n = ys.size(0)
    xT = torch.randn((n, in_channels, latent_size, latent_size), device=device)
    fn_vgg = None
    if not args.ldm_only:
        fn_vgg = lpips.LPIPS(net='vgg').to(device).eval().to(torch.bfloat16)

    # main training loop
    for epoch in range(args.epochs):
        for raw_image, latents, y in train_dataloader:
            raw_image = raw_image.to(device)
            labels = y.to(device)

            # extract the dinov2 (or other VF models) features
            with torch.no_grad():
                zs = []
                with accelerator.autocast():
                    for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                        raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                        z = encoder.forward_features(raw_image_)
                        if 'mocov3' in encoder_type: z = z = z[:, 1:] 
                        if 'dinov2' in encoder_type: z = z['x_norm_patchtokens']
                        zs.append(z)

            with accelerator.accumulate(model), accelerator.autocast():
                # NOTE: Set the model to training mode, but we fix the BN stats
                model.train()
                accelerator.unwrap_model(model).bn.eval()

                # 1. Sample the posterior from the latents distribution, normalization handled by BN of the model
                sampled_posterior = sample_posterior(latents, 1., 0.)

                # 2. Forward pass: SiT
                loss_kwargs = dict(
                    weighting=args.weighting,
                    path_type=args.path_type,
                    prediction=args.prediction,
                    align_only=False,
                    ldm_only=args.ldm_only,
                )
                model_kwargs = dict(
                    x=sampled_posterior,
                    y=labels,
                    zs=zs,
                    loss_kwargs=loss_kwargs,
                )
                if not args.ldm_only:
                    model_kwargs.update(
                        pixel=raw_image.float() / 127.5 - 1.,
                        vae=vae,
                        latents_scale=latents_scale,
                        latents_bias=latents_bias,
                        fn_vgg=fn_vgg,
                    )
                sit_outputs = model(**model_kwargs)

                # 3. Compute diffusion loss and REPA alignment loss, backpropagate the SiT loss, and update the model
                # sit_loss = sit_outputs["denoising_loss"].mean() + args.proj_coeff * sit_outputs["proj_loss"].mean() + sit_outputs["jit_loss"].mean()
                sit_loss = sit_outputs["denoising_loss"].mean() + args.proj_coeff * sit_outputs["proj_loss"].mean()
                if not args.ldm_only:
                    sit_loss = sit_loss + sit_outputs["jit_loss"].mean() + sit_outputs["lpips_loss_vgg"].mean()
                accelerator.backward(sit_loss)
                if accelerator.sync_gradients:
                    grad_norm_sit = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # 4. Update EMA
                if accelerator.sync_gradients:
                    unwrapped_model = accelerator.unwrap_model(model)
                    update_ema(ema, unwrapped_model._orig_mod if args.compile else unwrapped_model)

            ### enter
            if accelerator.sync_gradients:
                global_step += 1

                # Prepare the logs based on the current step
                logs = {
                    "sit_loss": accelerator.gather(sit_loss).mean().detach().item(),
                    "denoising_loss": accelerator.gather(sit_outputs["denoising_loss"]).mean().detach().item(),
                    "proj_loss": accelerator.gather(sit_outputs["proj_loss"]).mean().detach().item(),
                    "grad_norm_sit": accelerator.gather(grad_norm_sit).mean().detach().item(),
                    "epoch": epoch,
                }
                if not args.ldm_only:
                    logs.update({
                        "jit_loss": accelerator.gather(sit_outputs["jit_loss"]).mean().detach().item(),
                        "lpips_loss_vgg": accelerator.gather(sit_outputs["lpips_loss_vgg"]).mean().detach().item(),
                    })
                accelerator.log(logs, step=global_step)

                running_sit_loss += logs["sit_loss"]
                running_denoising_loss += logs["denoising_loss"]
                if not args.ldm_only:
                    running_jit_loss += logs["jit_loss"]
                    running_lpips_loss_vgg += logs["lpips_loss_vgg"]
                running_proj_loss += logs["proj_loss"]
                running_grad_norm += logs["grad_norm_sit"]
                running_steps += 1

                if (
                    log_interval > 0
                    and global_step % log_interval == 0
                    and accelerator.is_main_process
                    and running_steps > 0
                ):
                    avg_sit_loss = running_sit_loss / running_steps
                    avg_denoising_loss = running_denoising_loss / running_steps
                    avg_proj_loss = running_proj_loss / running_steps
                    avg_grad_norm = running_grad_norm / running_steps
                    log_msg = (
                        f"[Epoch {epoch} | Step {global_step}] "
                        f"sit_loss: {avg_sit_loss:.7f}, "
                        f"denoising_loss: {avg_denoising_loss:.7f}, "
                    )
                    if not args.ldm_only:
                        avg_jit_loss = running_jit_loss / running_steps
                        avg_lpips_loss_vgg = running_lpips_loss_vgg / running_steps
                        log_msg += (
                            f"jit_loss: {avg_jit_loss:.7f}, "
                            f"lpips_loss_vgg: {avg_lpips_loss_vgg:.7f}, "
                        )
                    log_msg += f"proj_loss: {avg_proj_loss:.7f}, grad_norm_sit: {avg_grad_norm:.7f}"
                    logger.info(log_msg)
                    running_sit_loss = 0.0
                    running_denoising_loss = 0.0
                    running_jit_loss = 0.0
                    running_lpips_loss_vgg = 0.0
                    running_proj_loss = 0.0
                    running_grad_norm = 0.0
                    running_steps = 0

            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    # `model` is wrapped by the `accelerator` object, so we need to unwrap them
                    unwrapped_model = accelerator.unwrap_model(model)

                    # model might be compiled, we extract the original model
                    original_model = unwrapped_model._orig_mod if args.compile else unwrapped_model

                    checkpoint = {
                        "model": original_model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            if (global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0)):
                # NOTE: Inference should use eval mode
                model.eval()
                with torch.no_grad():
                    unwrapped_model = accelerator.unwrap_model(model)
                    samples = euler_sampler(
                        unwrapped_model,
                        xT, 
                        ys,
                        num_steps=50, 
                        cfg_scale=4.0,
                        guidance_low=0.,
                        guidance_high=1.,
                        path_type=args.path_type,
                        heun=False,
                    ).to(torch.float32)
                    latents_stats = unwrapped_model.extract_latents_stats()
                    # latents_stats should be reshaped to [1, C, 1, 1]
                    latents_scale = latents_stats['latents_scale'].view(1, in_channels, 1, 1)
                    latents_bias = latents_stats['latents_bias'].view(1, in_channels, 1, 1)
                    samples = vae.decode(denormalize_latents(samples, latents_scale, latents_bias)).sample
                    samples = (samples + 1) / 2.
                out_samples = accelerator.gather(samples.to(torch.float32))
                accelerator.log({"samples": wandb.Image(array2grid(out_samples))})
                logging.info("Generating EMA samples done.")

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--sampling-steps", type=int, default=10000000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--continue-train-exp-dir", type=str, default=None)
    parser.add_argument("--wandb-history-path", type=str, default=None)

    # model
    parser.add_argument("--model", type=str, default="SiT-XL/1", choices=SiT_models.keys(),
                        help="The model to train, can be either from SiT or LightningDiT")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                        help="Whether to compile the model for faster training")

    # dataset
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--resolution", type=int, choices=[256], default=256)
    parser.add_argument("--batch-size", type=int, default=256)

    # precision
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=4)

    # loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"]) # currently we only support v-prediction
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    parser.add_argument("--proj-coeff", type=float, default=0.5)
    parser.add_argument("--weighting", default="uniform", type=str,
                        choices=["uniform", "lognormal"], help="Loss weihgting, uniform or lognormal")
    parser.add_argument("--ldm-only", "--first-stage-only", dest="ldm_only", action="store_true",
                        help="Run first-stage LDM-only training, matching REPA-E train_ldm_only.py. "
                             "Pixel branch modules are frozen and skipped.")

    # online eval
    parser.add_argument("--eval-epoch-interval", type=int, default=2,
                        help="Run full eval every N epochs; 0 disables eval")
    parser.add_argument("--eval-nproc-per-node", type=int, default=8)
    parser.add_argument("--eval-master-port", type=int, default=29501)
    parser.add_argument("--eval-num-fid-samples", type=int, default=50000)
    parser.add_argument("--eval-num-steps", type=int, default=50)
    parser.add_argument("--eval-cfg-scale", type=float, default=1.0)
    parser.add_argument("--eval-guidance-low", type=float, default=0.0)
    parser.add_argument("--eval-guidance-high", type=float, default=1.0)
    parser.add_argument("--eval-mode", type=str, default="sde")
    parser.add_argument("--eval-pproc-batch-size", type=int, default=32)
    parser.add_argument("--eval-label-sampling", type=str, default="equal", choices=["random", "equal"])
    parser.add_argument("--eval-sample-dir", type=str, default="samples")
    parser.add_argument("--eval-fid-script", type=str, default="calculate_fid.py")
    parser.add_argument("--eval-ref-npz", type=str, default="imagenet_in256_stats.npz")
    parser.add_argument("--eval-fid-batch-size", type=int, default=50)
    parser.add_argument("--eval-fid-dims", type=int, default=2048)
    parser.add_argument("--eval-fid-device", type=str, default="cuda")
    parser.add_argument("--eval-fid-num-workers", type=int, default=8)
    parser.add_argument("--eval-fid-sp-len", type=int, default=50000)
    parser.add_argument("--eval-cleanup-samples", action="store_true")

    parser.add_argument("--ignore-pixel-final-layer", action="store_true",
                        help="Skip loading pixel_final_layer from checkpoint")

    # vae
    parser.add_argument("--vae", type=str, default="f16d32", choices=["f8d4", "f16d32"])
    parser.add_argument("--vae-latents-name", type=str, default="e2e-vavae-400k",
                        help="The name of the pre-extracted latents")
    parser.add_argument("--vae-ckpt", type=str, default="pretrained/e2e-vavae-400k/e2e-vavae-400k.pt")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
