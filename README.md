# FAVDBench: Fine-grained Audible Video Description

**[OpenNLPLab](http://www.avlbench.opennlplab.cn/)**

[[`CVPR2023`]](https://openaccess.thecvf.com/content/CVPR2023/html/Shen_Fine-Grained_Audible_Video_Description_CVPR_2023_paper.html) [[`Project Page`]](http://www.avlbench.opennlplab.cn/papers/favd) [[`arXiv`]](https://arxiv.org/abs/2303.15616) [[`Demo`]](https://www.youtube.com/watch?v=iWJvTB-bTWk&ab_channel=OpenNLPLab)[[`BibTex`]](#Citation) [[`中文简介`]](https://mp.weixin.qq.com/s/_M57ZuOHH0UdwB6i9osqOA) 

This repository provides the official implementation for the CVPR2023 paper "Fine-grained Audible Video Description". 
We build a novel task: **FAVD** and a new dataset: **FAVDBench** in this paper.  

<p float="left">
  <img src="images/task_intro.png?raw=true" width="66.7%" />
</p>

## Apply for Dataset 

You can access the FAVDBench dataset by visiting the [OpenNLPLab/Download](http://www.avlbench.opennlplab.cn/download) webpage. To obtain the dataset, please complete the corresponding [Google Forms](https://forms.gle/5S3DWpBaV1UVczkf8). Once we receive your application, we will respond promptly. Alternatively, if you encounter any issues with the form, you can also submit your application (indicating your Name, Affiliation) via email to opennlplab@gmail.com.

* FAVDBench Dataset Google Forms: https://forms.gle/5S3DWpBaV1UVczkf8

These downloaded data can be placed or linked to the directory `AVLFormer/datasets`.


## Installation
In general, the code requires `python>=3.7`, as well as `pytorch>=1.10` and `torchvision>=0.8`. You can follow [recommend_env.sh]('https://github.com/OpenNLPLab/FAVDBench/blob/main/recommend_env.sh) to configure a recommend conda environment:
1. Install pytorch-related packages:
```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```
2. Install basic packages:
```bash
pip install fairscale opencv-python
pip install deepspeed PyYAML fvcore ete3 transformers pandas timm h5py
pip install tensorboardX easydict progressbar matplotlib future deprecated scipy av scikit-image boto3 einops addict yapf
```
3. Install mmcv-full
```bash
pip install mmcv-full==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
```

4. Install apex
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./
```
5. Clone related repo for eval 
```bash
cd ./AVLFormer/src/evalcap
git clone https://github.com/xiaoweihu/cider.git
git clone https://github.com/LuoweiZhou/coco-caption.git
mv ./coco-caption ./coco_caption 
```


## License


## Citation
If you use FAVD or FAVDBench in your research, please use the following BibTeX entry.

```
@InProceedings{Shen_2023_CVPR,
    author    = {Shen, Xuyang and Li, Dong and Zhou, Jinxing and Qin, Zhen and He, Bowen and Han, Xiaodong and Li, Aixuan and Dai, Yuchao and Kong, Lingpeng and Wang, Meng and Qiao, Yu and Zhong, Yiran},
    title     = {Fine-Grained Audible Video Description},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {10585-10596}
}
```