import argparse
import os

import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repae-ckpt", type=str, required=True,
                        help="Path to the REPA-E checkpoint")
    parser.add_argument("--vae-name", type=str, default="e2e-vae",
                        help="Name of the saved VAE model")
    parser.add_argument("--save-dir", type=str, default="exps",
                        help="Directory to save the extracted VAE weights")
    args = parser.parse_args()

    # Load the whole model ckpt
    ckpt = torch.load(args.repae_ckpt, map_location="cpu")

    # Extract the VAE weights
    vae_weights = ckpt["vae"]

    # Extract the VAE latent stats
    bn_running_mean = ckpt["ema"]["bn.running_mean"]
    bn_running_var = ckpt["ema"]["bn.running_var"]
    in_channels = bn_running_mean.shape[0]
    latents_bias = bn_running_mean.view(1, in_channels, 1, 1)
    latents_scale = bn_running_var.rsqrt().view(1, in_channels, 1, 1)
    latents_stats = {
        "latents_scale": latents_scale,
        "latents_bias": latents_bias,
    }

    # Extract the discriminator weights
    discriminator_weights = ckpt["discriminator"]

    # Finally, save the weights
    save_dir = os.path.join(args.save_dir, args.vae_name)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(vae_weights, os.path.join(save_dir, f"{args.vae_name}.pt"))
    torch.save(latents_stats, os.path.join(save_dir, f"{args.vae_name}-latents-stats.pt"))
    torch.save(discriminator_weights, os.path.join(save_dir, f"{args.vae_name}-discriminator-ckpt.pt"))
