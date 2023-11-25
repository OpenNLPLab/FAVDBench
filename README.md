# FAVDBench: Fine-grained Audible Video Description

<p align="center">
🏠 <a href="https://github.com/OpenNLPLab/FAVDBench" target="_blank">GitHub</a> •
🤗 <a href="https://huggingface.co/datasets/OpenNLPLab/FAVDBench" target="_blank">Hugging Face</a> •
🤖 <a href="https://openxlab.org.cn/datasets/OpenNLPLab/FAVDBench" target="_blank">OpenDataLab</a> •
💬 <a href="https://forms.gle/5S3DWpBaV1UVczkf8" target="_blank">Apply Dataset</a> 
</p>

[[`CVPR2023`]](https://openaccess.thecvf.com/content/CVPR2023/html/Shen_Fine-Grained_Audible_Video_Description_CVPR_2023_paper.html) [[`Project Page`]](http://www.avlbench.opennlplab.cn/papers/favd) [[`arXiv`]](https://arxiv.org/abs/2303.15616) [[`Demo`]](https://www.youtube.com/watch?v=iWJvTB-bTWk&ab_channel=OpenNLPLab)[[`BibTex`]](#Citation) [[`中文简介`]](https://mp.weixin.qq.com/s/_M57ZuOHH0UdwB6i9osqOA) 


This repository provides the official implementation for the CVPR2023 paper "Fine-grained Audible Video Description". 
We build a novel task: **FAVD** and a new dataset: **FAVDBench** in this paper.  

<p float="left">
  <img src="images/task_intro.png?raw=true" width="86.7%" />
</p>

:smiley: This repository offers the code necessary to replicate the results outlined in the paper. However, we **encourage** you to explore additional tasks using the **FAVDBench dataset**.

## Table of Content
- [FAVDBench: Fine-grained Audible Video Description](#favdbench-fine-grained-audible-video-description)
  - [Table of Content](#table-of-content)
  - [Apply for Dataset](#apply-for-dataset)
  - [Installation](#installation)
  - [Dataset Preparation](#dataset-preparation)
  - [Quick Links for Dataset Preparation](#quick-links-for-dataset-preparation)
  - [Experiments](#experiments)
    - [Preparation](#preparation)
    - [Training](#training)
    - [Inference](#inference)
  - [Quick Links for Experiments](#quick-links-for-experiments)
  - [Evaluation](#evaluation)
    - [AudioScore](#audioscore)
    - [EntitySore](#entitysore)
    - [CLIPScore](#clipscore)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)
  - [Citation](#citation)

## Apply for Dataset 

You can access the FAVDBench dataset by visiting the [OpenNLPLab/Download](http://www.avlbench.opennlplab.cn/download) webpage. To obtain the dataset, please complete the corresponding [Google Forms](https://forms.gle/5S3DWpBaV1UVczkf8). Once we receive your application, we will respond promptly. Alternatively, if you encounter any issues with the form, you can also submit your application (indicating your Name, Affiliation) via email to opennlplab@gmail.com.

* FAVDBench Dataset Google Forms: https://forms.gle/5S3DWpBaV1UVczkf8

These downloaded data can be placed or linked to the directory `AVLFormer/datasets`.


## Installation
In general, the code requires `python>=3.7`, as well as `pytorch>=1.10` and `torchvision>=0.8`. You can follow [`recommend_env.sh`]('https://github.com/OpenNLPLab/FAVDBench/blob/main/recommend_env.sh) to configure a recommend conda environment:
1. Create virtual env
    ```bash
    conda create -n FAVDBench; conda activate FAVDBench
    ```
2. Install pytorch-related packages:
    ```bash
    conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
    ```
3. Install basic packages:
    ```bash
    pip install fairscale opencv-python
    pip install deepspeed PyYAML fvcore ete3 transformers pandas timm h5py
    pip install tensorboardX easydict progressbar matplotlib future deprecated scipy av scikit-image boto3 einops addict yapf
    ```
4. Install mmcv-full
    ```bash
    pip install mmcv-full==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
    ```

5. Install apex
    ```bash
    git clone https://github.com/NVIDIA/apex
    cd apex
    pip install -v --disable-pip-version-check --no-cache-dir ./
    ```
    
6. Clone related repo for eval 
    ```bash
    cd ./AVLFormer/src/evalcap
    git clone https://github.com/xiaoweihu/cider.git
    git clone https://github.com/LuoweiZhou/coco-caption.git
    mv ./coco-caption ./coco_caption 
    ```

7. Install ffmpeg & ffprobe
  * Use `ffmpeg -version` and `ffprobe -version` to check whether ffmpeg and ffprobe are installed.
  * Installation guideline: 

    ```bash
      # For ubuntu
      sudo apt update
      sudo apt install ffmpeg

      # For mac
      brew update
      brew install ffmpeg
    ```
  

## Dataset Preparation
   
**📝Note:** 
* Please finish the above [installation](#installation) before the subsequent steps.   
* Check [Quick Links for Dataset Preparation](#quick-links-for-dataset-preparation) to download the processed files may help you to quickly enter the exp part.

---
1. Refer to the [Apply for Dataset](#apply-for-dataset) section to download the raw video files directly into the datasets folder.

2. Retrieve the [metadata.zip](https://github.com/OpenNLPLab/FAVDBench/releases/download/r-data-preparation/metadata.zip) file into the datasets folder, then proceed to unzip it.  

3. Activate conda env `conda activate FAVDBench`.

4. Extract the frames from videos and convert them into a single TSV (Tab-Separated Values) file.
    ```bash
    # check the path
    pwd
    >>> FAVDBench/AVLFormer

    # check the preparation
    ls datasets
    >>> audios metadata videos audios

    # data pre-processing
    bash data_prepro/run.sh

    # validate the data pre-processing
    ls datasets
    >>> audios frames  frame_tsv  metadata videos

    ls datasets/frames
    >>> train-32frames test-32frames val-32frames

    ls datasets/frame_tsv
    test_32frames.img.lineidx   test_32frames.img.tsv    test_32frames.img.lineidx.8b    
    val_32frames.img.lineidx    val_32frames.img.tsv     val_32frames.img.lineidx.8b
    train_32frames.img.lineidx  train_32frames.img.tsv   train_32frames.img.lineidx.8b       
    ```
 
    **📝Note**  
    * The contents within `datasets/frames` serve as intermediate files for training, although they hold utility for inference and scoring.
    * `datasets/frame_tsv` files are specifically designed for training purposes.
    * Should you encounter any problems, access [Quick Links for Dataset Preparation](#quick-links-for-dataset-preparation) to download the processed files or initiate a new issue in GitHub.

6. Convert the audio files in `mp3` format to the `h5py` format by archiving them.

    ```bash
    python data_prepro/convert_h5py.py train
    python data_prepro/convert_h5py.py val
    python data_prepro/convert_h5py.py test
    ```

    ```bash
    # check the preparation
    ls datasets/audio_hdf
    >>> test_mp3.hdf  train_mp3.hdf  val_mp3.hdf
    ```
    **📝Note**  
    * Should you encounter any problems, access [Quick Links for Dataset Preparation](#quick-links-for-dataset-preparation) to download the processed files or initiate a new issue in GitHub.

## Quick Links for Dataset Preparation 
|                |                                                                                             URL                                                                                             |              md5sum              |
| :------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------: |
| meta4raw-video |                                             📼 [meta.zip](https://github.com/OpenNLPLab/FAVDBench/releases/download/r-data-preparation/meta.zip)                                             | 5b50445f2e3136a83c95b396fc69c84a |
|    metadata    |                                        💻  [metadata.zip](https://github.com/OpenNLPLab/FAVDBench/releases/download/r-data-preparation/metadata.zip)                                         | f03e61e48212132bfd9589c2d8041cb1 |
|   audio_mp3    |                                        🎵 [audio_mp3.tar](https://github.com/OpenNLPLab/FAVDBench/releases/download/r-data-preparation/audio_mp3.tar)                                        | e2a3eb49edbb21273a4bad0abc32cda7 |
|   audio_hdf    |                                        🎵 [audio_hdf.tar](https://github.com/OpenNLPLab/FAVDBench/releases/download/r-data-preparation/audio_hdf.tar)                                        | 79f09f444ce891b858cb728d2fdcdc1b |
|   frame_tsv    | 🎆 [Dropbox](https://www.dropbox.com/scl/fi/7x1tvjqzapw64779ng8kp/frame_tsv.tar?rlkey=1mmnp67c265js8jpqr0s76js9&dl=0) / [百度网盘](https://pan.baidu.com/s/1PMqqdt3dqWniSZWeOcqK9A?pwd=h8ts) | 6c237a72d3a2bbb9d6b6d78ac1b55ba2 |


## Experiments

**📝Note:** 
* Please finish the above [installation](#installation) and [data preparation](#dataset-preparation) before the subsequent steps.   
* Check [Quick Links for Experiments](#quick-links-for-experiments) to download the pretrained weights may help your exps.

### Preparation
Please visit [Video Swin Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer#results-and-models) to download pre-trained weights models.

Download `swin_base_patch244_window877_kinetics400_22k.pth` and `swin_base_patch244_window877_kinetics600_22k.pth`, and place them under `models/video_swin_transformer` directory.
```bash
FAVDBench/AVLFormer
|-- datasets      (purposes)
|   |--audios     (raw-data)  
|   |--audio_hdf  (training, evaluation)
|   |--audio_mp3  (evaluation, inference)
|   |--frame_tsv  (training)
|   |--frames     (evaluation)
|   |--meta       (raw-data)
|   |--metadata   (training)
|   |--videos     (raw-data, inference)
|-- models  
|   |--captioning/bert-base-uncased
|   |-- video_swin_transformer
|    |   |-- swin_base_patch244_window877_kinetics600_22k.pth
|    |   |-- swin_base_patch244_window877_kinetics400_22k.pth
```

### Training
* The [run.sh](./AVLFormer/run.sh) file provides training scripts catered for single GPU, multiple GPUs, and distributed across multiple nodes with GPUs.
* The [hyperparameters](https://github.com/OpenNLPLab/FAVDBench/releases/download/r-pt-model/args.json) could be beneficial.
* For detailed instructions, refer to [this guide](./AVLFormer/README.md).

### Inference
*  The [inference.sh](./AVLFormer/inference.sh) file offers scripts for inferences.
*  **Attention**: The baseline for inference necessitates both raw video and audio data, which could be found [here](#quick-links-for-dataset-preparation).
*  For detailed instructions, refer to [this guide](./AVLFormer/README.md).


## Quick Links for Experiments
|                 |                                                                                URL                                                                                 |              md5sum              |
| :-------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------: |
|     weight      | 🔒  [GitHub](https://github.com/OpenNLPLab/FAVDBench/releases/download/r-pt-model/model.bin) / [百度网盘](https://pan.baidu.com/s/1fUu7i8PjEH2B_5gw5dFjXg?pwd=b9ns) | 5d6579198373b79a21cfa67958e9af83 |
| hyperparameters |                                   🧮  [args.json](https://github.com/OpenNLPLab/FAVDBench/releases/download/r-pt-model/args.json)                                   |                -                 |
|   prediction    |                    ☀️ [prediction_coco_fmt.json](https://github.com/OpenNLPLab/FAVDBench/releases/download/r-pt-model/prediction_coco_fmt.json)                     |                -                 |
|     metrics     |                                 🔢  [metrics.log](https://github.com/OpenNLPLab/FAVDBench/releases/download/r-pt-model/metrics.log)                                 |                -                 |


## Evaluation

### AudioScore
* For detailed instructions, refer to [this guide](./Metrics/AudioScore/README.md).

```bash
cd Metrics/AudioScore

python score.py \
    --pred_path PATH_to_PREDICTION_JSON_in_COCO_FORMAT \
```
**📝Note:** 
* Additional weights is requried to download, please refer to the [installation](./Metrics/AudioScore/README.md#installation).
* 
    |        |                                                 URL                                                 |              md5sum              |
    | :----: | :-------------------------------------------------------------------------------------------------: | :------------------------------: |
    | TriLip | 👍 [TriLip.bin](https://github.com/OpenNLPLab/FAVDBench/releases/download/r-metric-score/TriLip.bin) | 6baef8a9b383fa7c94a4c56856b0af6d |


### EntitySore
* For detailed instructions, refer to [this guide](./Metrics/EntityScore/README.md).

```bash
cd Metrics/EntityScore

python score.py \
    --pred_path PATH_to_PREDICTION_JSON_in_COCO_FORMAT \
    --refe_path AVLFormer/datasets/metadata/test.caption_coco_format.json \
    --model_path t5-base
```

### CLIPScore
* Please refer to [CLIPScore](https://github.com/jmhessel/clipscore) to evaluate the model.


## License
The community usage of FAVDBench model & code requires adherence to [Apache 2.0](https://github.com/OpenNLPLab/FAVDBench/blob/main/LICENSE). The FAVDBench model & code supports commercial use.


## Acknowledgments
Our project is developed based on the following open source projects:
- [SwinBERT](https://github.com/microsoft/SwinBERT) for the code baseline.
- [Video Swin Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer) for video model.
- [PaSST](https://github.com/kkoutini/PaSST) for audio model.
- [CLIP](https://github.com/openai/CLIP) for AudioScore model.


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
