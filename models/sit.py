# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import copy
import einops
from utils import denormalize_latents
from PIL import Image
def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################            
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        self.timestep_embedding = self.positional_embedding
        t_freq = self.timestep_embedding(t, dim=self.frequency_embedding_size).to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core SiT Model                                #
#################################################################################

class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=block_kwargs["qk_norm"]
            )
        if "fused_attn" in block_kwargs.keys():
            self.attn.fused_attn = block_kwargs["fused_attn"]
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
            )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)

        return x

class PixelFinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size,           # Transformer 输出维度 (如 1152)
        out_channels=3,        # RGB
        start_resolution=16,   # 起始分辨率 (16x16)
        mid_channels=256,      # CNN 中间通道数
    ):
        super().__init__()
        self.start_resolution = start_resolution
        self.pixel_patch_size = 4 
        self.final_dim = (self.pixel_patch_size ** 2) * out_channels
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.proj_in = nn.Linear(hidden_size, mid_channels)
        self.up_block = nn.Sequential(
            # Step 1: 16x16 -> 32x32
            nn.Conv2d(mid_channels, mid_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2), 
            nn.GroupNorm(num_groups=32, num_channels=mid_channels),
            nn.SiLU(),
            
            # Step 2: 32x32 -> 64x64
            nn.Conv2d(mid_channels, mid_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.GroupNorm(num_groups=32, num_channels=mid_channels),
            nn.SiLU(),
        )
        self.to_pixels = nn.Linear(mid_channels, self.final_dim)
        self.pixel_adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * self.final_dim, bias=True) 
        )

    def forward(self, x, c):
        """
        x: (N, 256, 1152) -> 对应 16x16 tokens
        c: (N, 1152)
        """
        N, T, D = x.shape
        H = W = int(T ** 0.5)
        x = self.norm_final(x)
        x = self.proj_in(x)
        x = einops.rearrange(x, 'n (h w) c -> n c h w', h=H, w=W) 
        x = self.up_block(x) 
        x = einops.rearrange(x, 'n c h w -> n (h w) c') # (N, 4096, 256)

        x_pixels = self.to_pixels(x) 

        mod = self.pixel_adaln(c).unsqueeze(1)
        shift, scale = mod.chunk(2, dim=-1)
        x_pixels = x_pixels * (1 + scale) + shift
        
        return x_pixels

class BottleneckPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Linear(pca_dim, embed_dim, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        #x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        x = self.proj1(x).flatten(2).transpose(1, 2)
        #x = self.norm(x)
        x = self.proj2(x)
        return x


class SiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        path_type='edm',
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        decoder_hidden_size=768,
        encoder_depth=8,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        z_dims=[768],
        projector_dim=2048,
        bn_momentum=0.1,
        time_thres=0.3,
        **block_kwargs # fused_attn
    ):
        super().__init__()
        self.path_type = path_type
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.z_dims = z_dims
        self.encoder_depth = encoder_depth
        self.time_thres = time_thres
        self.hidden_size = hidden_size  
        self.patch_size = patch_size

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        self.t_embedder = TimestepEmbedder(hidden_size) # timestep embedding type
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.pil_embedder = BottleneckPatchEmbed(256, 16, 3, 128, hidden_size, bias=True)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **block_kwargs) for _ in range(depth)
        ])
        self.projectors = nn.ModuleList([
            build_mlp(hidden_size, projector_dim, z_dim) for z_dim in z_dims
        ])
        self.final_layer = FinalLayer(decoder_hidden_size, patch_size, self.out_channels)
        # Note that we disable affine parameters in the batch norm layer, to avoid affine hacking diffusion loss
        self.bn = torch.nn.BatchNorm2d(
            in_channels, eps=1e-4, momentum=bn_momentum, affine=False, track_running_stats=True
        )
        self.bn.reset_running_stats()
        self.initialize_weights()
        self.add_net()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5)
            )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        w1 = self.pil_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.pil_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.pil_embedder.proj2.bias, 0)
    
    def add_net(self):
        self.pixel_final_layer = PixelFinalLayer(
            hidden_size=self.x_embedder.proj.weight.shape[0], # 1152
            out_channels=3,
            start_resolution=16, 
            mid_channels=256
        )
        nn.init.constant_(self.pixel_final_layer.pixel_adaln[-1].weight, 0)
        nn.init.constant_(self.pixel_final_layer.pixel_adaln[-1].bias, 0)
        nn.init.constant_(self.pixel_final_layer.to_pixels.weight, 0)
        nn.init.constant_(self.pixel_final_layer.to_pixels.bias, 0)


    def init_bn(self, latents_scale, latents_bias):
        self.bn.running_mean = latents_bias
        self.bn.running_var = (1. / latents_scale).pow(2)

    def extract_latents_stats(self):
        # rsqrt is the reciprocal of the square root
        latent_stats = dict(
            latents_scale=self.bn.running_var.rsqrt(),
            latents_bias=self.bn.running_mean,
        )
        return latent_stats

    def unpatchify(self, x, patch_size=None, out_channels=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels if out_channels is None else out_channels
        p = self.x_embedder.patch_size[0] if patch_size is None else patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def interpolant(self, t, path_type=None):
        if path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def forward(
        self,
        x,
        y,
        pixel=None,
        vae=None,
        latents_scale=None,
        latents_bias=None,
        fn_vgg=None,
        zs=None,
        loss_kwargs=None,
        time_input=None,
        noises=None,
    ):
        """
        Forward pass of SiT, integrating the loss function computation
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images UNNORMALIZED)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        zs: (N, L, C') tensor of external visual features
        loss_kwargs: dictionary of loss function arguments, should contain: `weighting`, `path_type`, `prediction`, `align_only`
        time_input: optionally provide a tensor of timesteps to use for the forward pass, otherwise sample from a distribution
        noises: optionally provide a tensor of noises to use for the forward pass, otherwise sample from a distribution
        """
        if loss_kwargs is None:
            raise ValueError("loss_kwargs must be provided.")
        if zs is None:
            raise ValueError("zs must be provided for REPA alignment.")
        pixel_branch_enabled = not loss_kwargs.get("ldm_only", False)

        # Normalize the input x with batch norm running stats
        normalized_x = self.bn(x)

        # sample timesteps if not provided
        if time_input is None:
            if loss_kwargs["weighting"] == "uniform":
                time_input = torch.rand((normalized_x.shape[0], 1, 1, 1))
            elif loss_kwargs["weighting"] == "lognormal":
                # sample timestep according to log-normal distribution of sigmas following EDM
                rnd_normal = torch.randn((normalized_x.shape[0], 1 ,1, 1))
                sigma = rnd_normal.exp()
                if loss_kwargs["path_type"] == "linear":
                    time_input = sigma / (1 + sigma)
                elif loss_kwargs["path_type"] == "cosine":
                    time_input = 2 / np.pi * torch.atan(sigma)
            else:
                raise NotImplementedError(f"Weighting scheme {loss_kwargs['weighting']} not implemented.")
        time_input = time_input.to(device=normalized_x.device, dtype=normalized_x.dtype)

        # sample noises if not provided
        if noises is None:
            noises = torch.randn_like(normalized_x)
        else:
            noises = noises.to(device=normalized_x.device, dtype=normalized_x.dtype)

        # compute interpolant
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input, path_type=loss_kwargs["path_type"])

        model_input = alpha_t * normalized_x + sigma_t * noises
        if loss_kwargs["prediction"] == 'v':
            model_target = d_alpha_t * normalized_x + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction

        x = self.x_embedder(model_input) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        if pixel_branch_enabled:
            if pixel is None:
                raise ValueError("pixel must be provided when pixel branch training is enabled.")
            tmask = (time_input.flatten() < self.time_thres).view(-1, 1, 1).to(dtype=x.dtype)
            pixel_noise = torch.randn_like(pixel) * 2
            pixel_input = alpha_t * pixel + sigma_t * pixel_noise
            pixel_emb = self.pil_embedder(pixel_input)

            pixel_keep_mask = torch.bernoulli(torch.full((x.shape[0], 1, 1), 0.5, device=x.device))
            pixel_emb = pixel_emb * pixel_keep_mask

            x = x + pixel_emb * tmask
        N, T, D = x.shape

        # timestep and class embedding
        t_embed = self.t_embedder(time_input.flatten())       # (N, D)
        y = self.y_embedder(y, self.training)                 # (N, D)
        c = t_embed + y                                       # (N, D)

        for i, block in enumerate(self.blocks):
            x = block(x, c)                                   # (N, T, D)
            if (i + 1) == self.encoder_depth:
                zs_tilde = [projector(x.reshape(-1, D)).reshape(N, T, -1) for projector in self.projectors]
                # NOTE: add a shortcut for feature extraction
                if loss_kwargs["align_only"]:
                    break
        x_1 = self.final_layer(x, c)                            # (N, T, patch_size ** 2 * out_channels)
        x_1 = self.unpatchify(x_1)                                # (N, out_channels, H, W)

        jit_loss = torch.tensor(0., device=x_1.device)
        lpips_loss_vgg = torch.tensor(0., device=x_1.device)

        if pixel_branch_enabled:
            if vae is None or latents_scale is None or latents_bias is None or fn_vgg is None:
                raise ValueError("vae, latents_scale, latents_bias, and fn_vgg are required for pixel branch training.")
            t_broadcast = time_input.view(x_1.shape[0], 1, 1, 1)
            x_decode = model_input - t_broadcast * x_1.detach()
            x_pixel_decode = vae.decode(
                denormalize_latents(
                    x_decode,
                    latents_scale.view(1, -1, 1, 1),
                    latents_bias.view(1, -1, 1, 1),
                ).float()
            ).sample
            x_2 = self.pixel_final_layer(x.detach(), c.detach())
            x_pixel = einops.rearrange(
                x_2,
                "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                h=64, w=64,
                p1=4, p2=4,
                c=3
            )
            x_pixel = x_pixel_decode + x_pixel
            jit_mask = (time_input.flatten() < 0.3).float().view(-1, 1, 1, 1)
            jit_loss = mean_flat((x_pixel - pixel) ** 2) * jit_mask.view(-1)
            lpips_loss_vgg = fn_vgg(x_pixel, pixel).view(-1) * jit_mask.view(-1)

        if loss_kwargs["align_only"]:
            denoising_loss = None
        else:
            denoising_loss = mean_flat((x_1 - model_target) ** 2)
            

        proj_loss = torch.tensor(0., device=x_1.device)
        bsz = zs[0].shape[0]
        for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
            for z_j, z_tilde_j in zip(z, z_tilde):
                z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
                z_j = torch.nn.functional.normalize(z_j, dim=-1) 
                proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
        proj_loss /= (len(zs) * bsz)

        return {
            "zs_tilde": zs_tilde,
            "model_output": x_1,
            "denoising_loss": denoising_loss,
            "jit_loss": jit_loss,
            "lpips_loss_vgg": lpips_loss_vgg,
            "proj_loss": proj_loss,
            "time_input": time_input,
            "noises": noises,
        }

    @torch.no_grad()
    def inference(self, x_input, t, y, pixel=None, vae=None, latents_scale=None, latents_bias=None):
        if pixel is not None and x_input.shape[0] != pixel.shape[0]:
            pixel = torch.cat([pixel] * 2, dim=0)
        x = self.x_embedder(x_input) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        if pixel is not None:
            tmask = (t.flatten() < self.time_thres).view(-1, 1, 1).to(dtype=x.dtype)
            pixel_emb = self.pil_embedder(pixel)
            x = x + pixel_emb * tmask
        # timestep and class embedding
        t_embed = self.t_embedder(t)             # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t_embed + y                          # (N, D)

        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x_1 = self.final_layer(x, c)                            # (N, T, patch_size ** 2 * out_channels)
        x_1 = self.unpatchify(x_1)                                # (N, out_channels, H, W)
        if pixel is None:
            return x_1

        # pixel prediction path
        if vae is None or latents_scale is None or latents_bias is None:
            raise ValueError("vae, latents_scale, and latents_bias are required when pixel is provided.")
        t_broadcast = t.view(x_1.shape[0], 1, 1, 1)
        x_decode = x_input - t_broadcast * x_1.detach()
        x_pixel_decode = vae.decode(denormalize_latents(x_decode, latents_scale.view(1, -1, 1, 1), latents_bias.view(1, -1, 1, 1)).float()).sample
        # pixel_emb_decode = self.pil_embedder(x_pixel_decode)
        # x_pixel_tokens = x + pixel_emb_decode
        
        x_2 = self.pixel_final_layer(x.detach(), c.detach())
        x_pixel = einops.rearrange(
            x_2, 
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", 
            h=64, w=64, 
            p1=4, p2=4, 
            c=3
        )
        x_pixel = x_pixel_decode + x_pixel
        return x_1, x_pixel

    @torch.no_grad()
    def forward_feats(self, x, t, y, depth):
        assert 1 <= depth <= len(self.blocks)
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        # timestep and class embedding
        t_embed = self.t_embedder(t)             # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t_embed + y                          # (N, D)

        for i, block in enumerate(self.blocks):
            x = block(x, c)                      # (N, T, D)
            if (i + 1) == depth:
                return x

        return None


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   SiT Configs                                  #
#################################################################################

def SiT_XL_1(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=1, num_heads=16, **kwargs)

def SiT_XL_2(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def SiT_XL_4(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def SiT_XL_8(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def SiT_L_1(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=1, num_heads=16, **kwargs)

def SiT_L_2(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def SiT_L_4(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def SiT_L_8(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def SiT_B_1(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=1, num_heads=12, **kwargs)

def SiT_B_2(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def SiT_B_4(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def SiT_B_8(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def SiT_S_1(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=1, num_heads=6, **kwargs)

def SiT_S_2(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def SiT_S_4(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def SiT_S_8(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


SiT_models = {
    'SiT-XL/1': SiT_XL_1,  'SiT-XL/2': SiT_XL_2,  'SiT-XL/4': SiT_XL_4,  'SiT-XL/8': SiT_XL_8,
    'SiT-L/1': SiT_L_1,  'SiT-L/2':  SiT_L_2,   'SiT-L/4':  SiT_L_4,   'SiT-L/8':  SiT_L_8,
    'SiT-B/1': SiT_B_1,  'SiT-B/2':  SiT_B_2,   'SiT-B/4':  SiT_B_4,   'SiT-B/8':  SiT_B_8,
    'SiT-S/1': SiT_S_1,  'SiT-S/2':  SiT_S_2,   'SiT-S/4':  SiT_S_4,   'SiT-S/8':  SiT_S_8,
}
