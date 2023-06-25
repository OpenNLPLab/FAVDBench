# create virtual env
conda create -n FAVDBench
# activate virtual env
conda activate FAVDBench

# install packages
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install fairscale opencv-python
pip install deepspeed PyYAML fvcore ete3 transformers pandas timm h5py
pip install tensorboardX easydict progressbar matplotlib future deprecated scipy av scikit-image boto3 einops addict yapf

# install mmcv
pip install mmcv-full==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html

# install apex
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./

# clone related repo for eval
cd ./AVLFormer/src/evalcap
git clone https://github.com/xiaoweihu/cider.git
git clone https://github.com/LuoweiZhou/coco-caption.git
mv ./coco-caption ./coco_caption 