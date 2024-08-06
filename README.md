Youu can using Diffree in ComfyUI 
---
Diffree: Text-Guided Shape Free Object Inpainting with Diffusion Model

"Diffree" From: [Diffree](https://github.com/OpenGVLab/Diffree)
----

1.Installation
-----
  In the ./ComfyUI /custom_node directory, run the following:   

  ``` python 
  git clone https://github.com/smthemex/ComfyUI_Diffree.git

  ```
2.requirements  
----

if miss K_diffusion...   

```
pip install -r requirements.txt

```
如果是便携包的，提示缺少K-diffusion需要在comfyUI/python_embeded目录下，打开CMD pip安装需求文件;    
if using portable standalone build for Windows 'comfyUI, you need in comfyUI/python_embeded;   

```
python -m pip install -r requirements.txt

```
或者 or: python pip install -r requirements.txt --target "you path/comfyUI/python_embeded/Lib/site-packages"  

Based on the SD model, ComfyUI users basically do not need to install any requirement libraries。  
基于sd的模型，comfyUI用户，基本上不用装任何需求库  

If a module is missing, please open 'nou_need-requirements.txt'  
如果缺失库，请打开nou_need_requirements.txt文件看你少了啥

3 Need  model 
----

3.1 base model，只有1个，only a model:     [diffree-step=000010999.ckpt](https://huggingface.co/LiruiZhao/Diffree/tree/main) 

```
├── ComfyUI/models/checkpoints/
|      ├── diffree-step=000010999.ckpt
```

3.2 vae  
you can try some one

4 using tips
---
--推荐使用512或者类似的尺寸，他们用的256图幅训练，推荐的是320（稍微小了，现在SDXL当道）;   
--Recommend using a size of 512 or similar, they use 256 frames for training, and the recommended size is 320;       

5 Example
----
 Text-Guided Shape Free Object Inpainting  常规文生内绘图演示。    
![](https://github.com/smthemex/ComfyUI_Diffree/blob/main/example/example.png)


My ComfyUI node list：
-----
1、ParlerTTS node:[ComfyUI_ParlerTTS](https://github.com/smthemex/ComfyUI_ParlerTTS)     
2、Llama3_8B node:[ComfyUI_Llama3_8B](https://github.com/smthemex/ComfyUI_Llama3_8B)      
3、HiDiffusion node：[ComfyUI_HiDiffusion_Pro](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro)   
4、ID_Animator node： [ComfyUI_ID_Animator](https://github.com/smthemex/ComfyUI_ID_Animator)       
5、StoryDiffusion node：[ComfyUI_StoryDiffusion](https://github.com/smthemex/ComfyUI_StoryDiffusion)  
6、Pops node：[ComfyUI_Pops](https://github.com/smthemex/ComfyUI_Pops)   
7、stable-audio-open-1.0 node ：[ComfyUI_StableAudio_Open](https://github.com/smthemex/ComfyUI_StableAudio_Open)        
8、GLM4 node：[ComfyUI_ChatGLM_API](https://github.com/smthemex/ComfyUI_ChatGLM_API)   
9、CustomNet node：[ComfyUI_CustomNet](https://github.com/smthemex/ComfyUI_CustomNet)           
10、Pipeline_Tool node :[ComfyUI_Pipeline_Tool](https://github.com/smthemex/ComfyUI_Pipeline_Tool)    
11、Pic2Story node :[ComfyUI_Pic2Story](https://github.com/smthemex/ComfyUI_Pic2Story)   
12、PBR_Maker node:[ComfyUI_PBR_Maker](https://github.com/smthemex/ComfyUI_PBR_Maker)      
13、ComfyUI_Streamv2v_Plus node:[ComfyUI_Streamv2v_Plus](https://github.com/smthemex/ComfyUI_Streamv2v_Plus)   
14、ComfyUI_MS_Diffusion node:[ComfyUI_MS_Diffusion](https://github.com/smthemex/ComfyUI_MS_Diffusion)   
15、ComfyUI_AnyDoor node: [ComfyUI_AnyDoor](https://github.com/smthemex/ComfyUI_AnyDoor)  
16、ComfyUI_Stable_Makeup node: [ComfyUI_Stable_Makeup](https://github.com/smthemex/ComfyUI_Stable_Makeup)  
17、ComfyUI_EchoMimic node:  [ComfyUI_EchoMimic](https://github.com/smthemex/ComfyUI_EchoMimic)   
18、ComfyUI_FollowYourEmoji node: [ComfyUI_FollowYourEmoji](https://github.com/smthemex/ComfyUI_FollowYourEmoji)   
19、ComfyUI_Diffree node: [ComfyUI_Diffree](https://github.com/smthemex/ComfyUI_Diffree)  


Citation
------
OpenGVLab/Diffree
```
@article{zhao2024diffree,
  title={Diffree: Text-Guided Shape Free Object Inpainting with Diffusion Model},
  author={Zhao, Lirui and Yang, Tianshuo and Shao, Wenqi and Zhang, Yuxin and Qiao, Yu and Luo, Ping and Zhang, Kaipeng and Ji, Rongrong},
  journal={arXiv preprint arXiv:2407.16982},
  year={2024}
}
```
