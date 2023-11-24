## Quick Links for Dataset Preparation 
|                |                                                                                             URL                                                                                             |              md5sum              |
| :------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------: |
| meta4raw-video |                                             📼 [meta.zip](https://github.com/OpenNLPLab/FAVDBench/releases/download/r-data-preparation/meta.zip)                                             | 5b50445f2e3136a83c95b396fc69c84a |
|    metadata    |                                        💻  [metadata.zip](https://github.com/OpenNLPLab/FAVDBench/releases/download/r-data-preparation/metadata.zip)                                         | f03e61e48212132bfd9589c2d8041cb1 |
|   audio_mp3    |                                        🎵 [audio_mp3.tar](https://github.com/OpenNLPLab/FAVDBench/releases/download/r-data-preparation/audio_mp3.tar)                                        | e2a3eb49edbb21273a4bad0abc32cda7 |
|   audio_hdf    |                                        🎵 [audio_hdf.tar](https://github.com/OpenNLPLab/FAVDBench/releases/download/r-data-preparation/audio_hdf.tar)                                        | 79f09f444ce891b858cb728d2fdcdc1b |
|   frame_tsv    | 🎆 [Dropbox](https://www.dropbox.com/scl/fi/7x1tvjqzapw64779ng8kp/frame_tsv.tar?rlkey=1mmnp67c265js8jpqr0s76js9&dl=0) / [百度网盘](https://pan.baidu.com/s/1PMqqdt3dqWniSZWeOcqK9A?pwd=h8ts) | 6c237a72d3a2bbb9d6b6d78ac1b55ba2 |

## Directory Overview
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