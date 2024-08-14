# !/usr/bin/env python
# -*- coding: UTF-8 -*-

import math
import random
import sys
import os
from tqdm.auto import trange
import einops
from . import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps, ImageFilter
from torch import autocast
import cv2

from .stable_diffusion.ldmx.util import instantiate_from_config
import folder_paths
from comfy.utils import common_upscale

cur_path = os.path.dirname(os.path.abspath(__file__))
MAX_SEED = np.iinfo(np.int32).max
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

def get_instance_path(path):
    instance_path = os.path.normpath(path)
    if sys.platform == 'win32':
        instance_path = instance_path.replace('\\', "/")
    return instance_path

def pil2tensor(pil_img):
    tensor = torch.from_numpy(np.array(pil_img).astype(np.float32) / 255.0).unsqueeze(0)
    return tensor

def tensor2phi(tensor):
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    pil_img = Image.fromarray(image_np, mode='RGB')
    return pil_img

def nomarl_upscale(img_tensor, width, height):
    samples = img_tensor.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    return samples

def nomarl_upscale_to_pil(img_tensor, width, height):
    samples = img_tensor.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    img_pil = tensor2phi(samples)
    return img_pil

def narry_list_to_pill_ist(narry_list_in):
    for i in range(len(narry_list_in)):
        value = narry_list_in[i]
        modified_value = pil2tensor(value)
        narry_list_in[i] = modified_value
    return narry_list_in

def narry_list(list_in):
    for i in range(len(list_in)):
        value = list_in[i]
        modified_value = pil2tensor(value)
        list_in[i] = modified_value
    return list_in

def phi_list(list_in):
    for i in range(len(list_in)):
        value = list_in[i]
        list_in[i] = value
    return list_in

def get_local_path(file_path, model_name,model_dir):
    path = os.path.join(file_path, "models", model_name, model_dir)
    model_path = os.path.normpath(path)
    if sys.platform.startswith('win32'):
        model_path = model_path.replace('\\', "/")
    return model_path

def get_image_path_list(folder_name):
    image_basename_list = os.listdir(folder_name)
    image_path_list = sorted(
        [os.path.join(folder_name, basename) for basename in image_basename_list]
    )
    return image_path_list

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z_0, z_1, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z_0 = einops.repeat(z_0, "1 ... -> n ...", n=3)
        cfg_z_1 = einops.repeat(z_1, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        output_0, output_1 = self.inner_model(cfg_z_0, cfg_z_1, cfg_sigma, cond=cfg_cond)
        out_cond_0, out_img_cond_0, out_uncond_0 = output_0.chunk(3)
        out_cond_1, _, _ = output_1.chunk(3)
        return out_uncond_0 + text_cfg_scale * (out_cond_0 - out_img_cond_0) + image_cfg_scale * (out_img_cond_0 - out_uncond_0), \
            out_cond_1

def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]


class CompVisDenoiser(K.external.CompVisDenoiser):
    def __init__(self, model, quantize=False, device='cpu'):
        super().__init__(model, quantize, device)
    
    def get_eps(self, *args, **kwargs):
        return self.inner_model.apply_model(*args, **kwargs)
    
    def forward(self, input_0, input_1, sigma, **kwargs):
        c_out, c_in = [append_dims(x, input_0.ndim) for x in self.get_scalings(sigma)]
        # eps_0, eps_1 = self.get_eps(input_0 * c_in, input_1 * c_in, self.sigma_to_t(sigma), **kwargs)
        eps_0, eps_1 = self.get_eps(input_0 * c_in, self.sigma_to_t(sigma).cuda(), **kwargs)
        
        return input_0 + eps_0 * c_out, eps_1


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


def default_noise_sampler(x):
    return lambda sigma, sigma_next: torch.randn_like(x)


def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    if not eta:
        return sigma_to, 0.
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up


def decode_mask(mask, height=256, width=256):
    mask = nn.functional.interpolate(mask, size=(height, width), mode="bilinear", align_corners=False)
    mask = torch.where(mask > 0, 1, -1)  # Thresholding step
    mask = torch.clamp((mask + 1.0) / 2.0, min=0.0, max=1.0)
    mask = 255.0 * rearrange(mask, "1 c h w -> h w c")
    mask = torch.cat([mask, mask, mask], dim=-1)
    mask = mask.type(torch.uint8).cpu().numpy()
    return mask


def sample_euler_ancestral(model, x_0, x_1, sigmas, height, width, extra_args=None, disable=None, eta=1., s_noise=1.,
                           noise_sampler=None):
    """Ancestral sampling with Euler method steps."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x_0) if noise_sampler is None else noise_sampler
    s_in = x_0.new_ones([x_0.shape[0]])
    
    mask_list = []
    image_list = []
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised_0, denoised_1 = model(x_0, x_1, sigmas[i] * s_in, **extra_args)
        image_list.append(denoised_0)
        
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        d_0 = to_d(x_0, sigmas[i], denoised_0)
        
        # Euler method
        dt = sigma_down - sigmas[i]
        x_0 = x_0 + d_0 * dt
        
        if sigmas[i + 1] > 0:
            x_0 = x_0 + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
        
        x_1 = denoised_1
        mask_list.append(decode_mask(x_1, height, width))
    
    image_list = torch.cat(image_list, dim=0)
    
    return x_0, x_1, image_list, mask_list


class Diffree_Model_Loader:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (["none"] + folder_paths.get_filename_list("checkpoints"),),
                "vae": (["none"] + folder_paths.get_filename_list("vae"),),
            }
        }
    
    RETURN_TYPES = ("MODEL","MODEL","MODEL", )
    RETURN_NAMES = ("model", "model_wrap","model_wrap_cfg",)
    FUNCTION = "main_loader"
    CATEGORY = "Diffree"
    
    def main_loader(self, ckpt_name, vae,):
        if vae != "none":
            vae_ckpt = folder_paths.get_full_path("vae", vae)
        else:
            vae_ckpt = None
        ckpt = folder_paths.get_full_path("checkpoints", ckpt_name)
        config_path = os.path.join(cur_path, "config", "generate.yaml")
        config = OmegaConf.load(config_path)
        model = load_model_from_config(config, ckpt, vae_ckpt)
        model.eval().cuda()
        model_wrap = CompVisDenoiser(model)
        model_wrap_cfg = CFGDenoiser(model_wrap)
        return (model,model_wrap,model_wrap_cfg,)

@torch.no_grad()
def generate(
        model,
        model_wrap,
        model_wrap_cfg,
        image: Image.Image,
        instruction: str,
        width: int,
        height: int,
        steps: int,
        randomize_seed: bool,
        seed: int,
        randomize_cfg: bool,
        text_cfg_scale: float,
        image_cfg_scale: float,
        rgb_mode: str,
):

    null_token = model.get_learned_conditioning([""])
    seed = random.randint(0, 100000) if randomize_seed else seed
    text_cfg_scale = round(random.uniform(6.0, 9.0), ndigits=2) if randomize_cfg else text_cfg_scale
    image_cfg_scale = round(random.uniform(1.2, 1.8), ndigits=2) if randomize_cfg else image_cfg_scale
    
    input_image_copy = image
    
    if instruction == "":
        instruction="reflective sunglasses" # prompt为空时，返回默认prompt
    
    model.cuda()
    with torch.no_grad(), autocast("cuda"), model.ema_scope():
        cond = {}
        cond["c_crossattn"] = [model.get_learned_conditioning([instruction]).to(model.device)]
        input_image = 2 * torch.tensor(np.array(image)).float() / 255 - 1
        input_image = rearrange(input_image, "h w c -> 1 c h w").to(model.device)
        cond["c_concat"] = [model.encode_first_stage(input_image).mode().to(model.device)]
        
        uncond = {}
        uncond["c_crossattn"] = [null_token.to(model.device)]
        uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]
        
        sigmas = model_wrap.get_sigmas(steps).to(model.device)
        
        extra_args = {
            "cond": cond,
            "uncond": uncond,
            "text_cfg_scale": text_cfg_scale,
            "image_cfg_scale": image_cfg_scale,
        }
        torch.manual_seed(seed)
        z_0 = torch.randn_like(cond["c_concat"][0]).to(model.device) * sigmas[0]
        z_1 = torch.randn_like(cond["c_concat"][0]).to(model.device) * sigmas[0]
        
        z_0, z_1, image_list, mask_list = sample_euler_ancestral(model_wrap_cfg, z_0, z_1, sigmas, height, width,
                                                                 extra_args=extra_args)
        
        x_0 = model.decode_first_stage(z_0)
        
        x_1 = nn.functional.interpolate(z_1, size=(height, width), mode="bilinear", align_corners=False)
        x_1 = torch.where(x_1 > 0, 1, -1)  # Thresholding step
        x_0 = torch.clamp((x_0 + 1.0) / 2.0, min=0.0, max=1.0)
        x_1 = torch.clamp((x_1 + 1.0) / 2.0, min=0.0, max=1.0)
        x_0 = 255.0 * rearrange(x_0, "1 c h w -> h w c")
        x_1 = 255.0 * rearrange(x_1, "1 c h w -> h w c")
        x_1 = torch.cat([x_1, x_1, x_1], dim=-1)
        edited_image = Image.fromarray(x_0.type(torch.uint8).cpu().numpy())
        edited_mask = Image.fromarray(x_1.type(torch.uint8).cpu().numpy())
        
        
        # 对edited_mask做膨胀
        edited_mask_copy = edited_mask.copy()
        kernel = np.ones((3, 3), np.uint8)
        edited_mask = cv2.dilate(np.array(edited_mask), kernel, iterations=3)
        edited_mask = Image.fromarray(edited_mask)
        
        m_img = edited_mask.filter(ImageFilter.GaussianBlur(radius=3))
        m_img = np.asarray(m_img).astype('float') / 255.0
        img_np = np.asarray(input_image_copy).astype('float') / 255.0
        ours_np = np.asarray(edited_image).astype('float') / 255.0
        #print(ours_np)
        mix_image_np = m_img * ours_np + (1 - m_img) * img_np
        mix_image = Image.fromarray((mix_image_np * 255).astype(np.uint8)).convert('RGB')
        
        rgb_choice = np.array(mix_image).astype('float') * 1
        if rgb_mode=="red":
            rgb_choice[:, :, 0] = 180.0
            rgb_choice[:, :, 2] = 0
            rgb_choice[:, :, 1] = 0
        elif rgb_mode=="blue":
            rgb_choice[:, :, 0] = 0
            rgb_choice[:, :, 2] = 180.0
            rgb_choice[:, :, 1] = 0
        else:# green
            rgb_choice[:, :, 0] = 0
            rgb_choice[:, :, 2] = 0
            rgb_choice[:, :, 1] = 180.0
            
        
        mix_result_with_rgb_mask = np.array(mix_image)
        mix_result_with_rgb_mask = Image.fromarray(
            (mix_result_with_rgb_mask.astype('float') * (1 - m_img.astype('float') / 2.0) +
             m_img.astype('float') / 2.0 * rgb_choice).astype('uint8'))
        
        mask_img=Image.fromarray((m_img * 255).astype(np.uint8)).convert('RGB')
        
        return mix_image, mask_img,mix_result_with_rgb_mask


class Diffree_Sampler:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "model_wrap": ("MODEL",),
                "model_wrap_cfg": ("MODEL",),
                "prompt": ("STRING", {"multiline": True, "default": "reflective sunglasses"}),
                "randomize_seed":("BOOLEAN", {"default": False},),
                "seed": ("INT", {"default": 100, "min": 0, "max": MAX_SEED}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "randomize_cfg": ("BOOLEAN", {"default": False},),
                "text_cfg": ("FLOAT", {"default": 7.5, "min": 0.1, "max": 10.0, "step": 0.1, "round": 0.01}),
                "img_cfg": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 10.0, "step": 0.1, "round": 0.01}),
                "width": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64, "display": "number"}),
                "height": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64, "display": "number"}),
                "rgb_mode":(["red","green","blue",],),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1, }),}
        }
  
    RETURN_TYPES = ("IMAGE","IMAGE","IMAGE",)
    RETURN_NAMES = ("image","mask","rgb_mask",)
    FUNCTION = "df_sampler"
    CATEGORY = "Diffree"
    
    def df_sampler(self, image, model,model_wrap, model_wrap_cfg, prompt, randomize_seed,seed, steps,randomize_cfg, text_cfg,img_cfg,width,height,rgb_mode,batch_size,):
        image=nomarl_upscale_to_pil(image,width,height)
        mix_image_list,mask_img_list,mix_result_with_rgb_mask_list=[],[],[]
        for i in range(batch_size):
            mix_image,mask_img,mix_result_with_rgb_mask=generate(model,model_wrap,model_wrap_cfg,image,prompt,width,height,steps,randomize_seed,seed,randomize_cfg,text_cfg,img_cfg,rgb_mode)
            mix_image_list.append(mix_image)
            mask_img_list.append(mask_img)
            mix_result_with_rgb_mask_list.append(mix_result_with_rgb_mask)

        mix_image=narry_list(mix_image_list)
        mask_img = narry_list(mask_img_list)
        mix_result_with_rgb_mask = narry_list(mix_result_with_rgb_mask_list)

        image=torch.from_numpy(np.fromiter(mix_image, np.dtype((np.float32, (height, width, 3)))))
        mask = torch.from_numpy(np.fromiter(mask_img, np.dtype((np.float32, (height, width, 3)))))
        rgb_mask=torch.from_numpy(np.fromiter(mix_result_with_rgb_mask, np.dtype((np.float32, (height, width, 3)))))
        
        return (image,mask,rgb_mask)


NODE_CLASS_MAPPINGS = {
    "Diffree_Model_Loader": Diffree_Model_Loader,
    "Diffree_Sampler": Diffree_Sampler,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Diffree_Model_Loader": "Diffree_Model_Loader",
    "Diffree_Sampler": "Diffree_Sampler",

}
