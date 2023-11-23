## Dataset Preparation
   
**📝Note:** 
* Please finish the above [installation](../../README.md#installation) before the subsequent steps.   
* Check [Quick Links for Dataset Preparation](../../README.md#quick-links-for-dataset-preparation) to download the processed files may help you to quickly enter the exp part.

---
1. Refer to the [Apply for Dataset](../../README.md#apply-for-dataset) section to download the raw video files directly into the datasets folder.

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
    * Should you encounter any problems, access [Quick Links for Dataset Preparation](../../README.md#quick-links-for-dataset-preparation) to download the processed files or initiate a new issue in GitHub.

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
    * Should you encounter any problems, access [Quick Links for Dataset Preparation](../../README.md#quick-links-for-dataset-preparation) to download the processed files or initiate a new issue in GitHub.