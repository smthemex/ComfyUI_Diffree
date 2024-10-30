# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import numpy as np
import torch
from omegaconf import OmegaConf
from .utils import load_model_from_config,CompVisDenoiser,CFGDenoiser,nomarl_upscale_to_pil,generate,load_images
import folder_paths

cur_path = os.path.dirname(os.path.abspath(__file__))
MAX_SEED = np.iinfo(np.int32).max
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
dtype = torch.float16 if device == "cuda" else torch.float
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
    
    RETURN_TYPES = ("DIFFREE_MODEL", )
    RETURN_NAMES = ("pipe",)
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
        model.eval()
        model_wrap = CompVisDenoiser(model)
        model_wrap_cfg = CFGDenoiser(model_wrap)
        pipe={"model":model,"model_wrap":model_wrap,"model_wrap_cfg":model_wrap_cfg}
        return (pipe,)

class Diffree_Sampler:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pipe": ("DIFFREE_MODEL",),
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
    
    def df_sampler(self, image, pipe,prompt, randomize_seed,seed, steps,randomize_cfg, text_cfg,img_cfg,width,height,rgb_mode,batch_size,):
        
        model=pipe.get("model")
        model_wrap=pipe.get("model_wrap")
        model_wrap_cfg=pipe.get("model_wrap_cfg")
        
        model.cuda() if model is not None else model
        model_wrap_cfg.cuda() if model_wrap_cfg is not None else model_wrap_cfg
        
        image=nomarl_upscale_to_pil(image,width,height)
        mix_image_list,mask_img_list,mix_result_with_rgb_mask_list=[],[],[]
        for i in range(batch_size):
            mix_image,mask_img,mix_result_with_rgb_mask=generate(model,model_wrap,model_wrap_cfg,image,prompt,width,height,steps,randomize_seed,seed,randomize_cfg,text_cfg,img_cfg,rgb_mode,dtype)
            mix_image_list.append(mix_image)
            mask_img_list.append(mask_img)
            mix_result_with_rgb_mask_list.append(mix_result_with_rgb_mask)
            
        image=load_images(mix_image_list)
        mask = load_images(mask_img_list)
        rgb_mask=load_images(mix_result_with_rgb_mask_list)
        model.cpu()
        model_wrap.cpu()
        model_wrap_cfg.inner_model.cpu()
        torch.cuda.empty_cache()
        return (image,mask,rgb_mask)


NODE_CLASS_MAPPINGS = {
    "Diffree_Model_Loader": Diffree_Model_Loader,
    "Diffree_Sampler": Diffree_Sampler,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Diffree_Model_Loader": "Diffree_Model_Loader",
    "Diffree_Sampler": "Diffree_Sampler",

}
