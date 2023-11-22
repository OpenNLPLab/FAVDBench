echo 'TRAIN'

python ./data_prepro/extract_frames.py \
--video_root_dir datasets/videos \
--save_dir datasets/frames/train- \
--video_info_tsv datasets/metadata/train.img.tsv \
--num_frames 32 \

python ./data_prepro/create_image_frame_tsv.py \
--dataset FAVD \
--split train \
--image_size 256 \
--num_frames 32 \


echo 'VAL'

python ./data_prepro/extract_frames.py \
--video_root_dir datasets/videos \
--save_dir datasets/frames/val- \
--video_info_tsv datasets/metadata/val.img.tsv \
--num_frames 32 \

python ./data_prepro/create_image_frame_tsv.py \
--dataset FAVD \
--split val \
--image_size 256 \
--num_frames 32 \


echo 'TEST'

python ./data_prepro/extract_frames.py \
--video_root_dir datasets/videos \
--save_dir datasets/frames/test- \
--video_info_tsv datasets/metadata/test.img.tsv \
--num_frames 32 \

python ./data_prepro/create_image_frame_tsv.py \
--dataset FAVD \
--split test \
--image_size 256 \
--num_frames 32 \

