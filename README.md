Youu can using Diffree in ComfyUI 
---
[Diffree](https://github.com/OpenGVLab/Diffree/tree/main): Text-Guided Shape Free Object Inpainting with Diffusion Model

----

Update
----
**2024/10/30**
* 优化一些代码，但是还有4G左右的占用一直没有被清理，有空再搞吧。因为推理用的autocast("cuda")，所以有些较旧的cuda卡可能会OOM


1.Installation
-----
  In the ./ComfyUI /custom_node directory, run the following:   

  ``` python 
  git clone https://github.com/smthemex/ComfyUI_Diffree.git

  ```
2.requirements  
----

if  K_diffusion...  missing  module...
check “no_need_requirements.txt”,pip missing module.
K_diffusion 需求的库文件也在no_need_requirements.txt 里，缺啥装啥

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
基于sd的模型，comfyUI安装版的用户，基本上不用装任何需求库  

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
模型内置，并不需要 you can try some one

4 using tips
---
--推荐使用512或者类似的尺寸，他们用的256图幅训练，推荐的是320（稍微小了，现在SDXL当道）;   
--Recommend using a size of 512 or similar, they use 256 frames for training, and the recommended size is 320;       

5 Example
----
 Text-Guided Shape Free Object Inpainting  常规文生内绘图演示。    
![](https://github.com/smthemex/ComfyUI_Diffree/blob/main/example.png)


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
