# FAVDBench: Fine-grained Audible Video Description
- [FAVDBench: Fine-grained Audible Video Description](#favdbench-fine-grained-audible-video-description)
  - [Evaluation](#evaluation)
    - [AudioScore](#audioscore)
    - [EntitySore](#entitysore)
    - [CLIPScore](#clipscore)
  - [COCO Format](#coco-format)

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

## COCO Format
```json
[
    {
        "image_id": xxx,
        "caption": xxx
    },
    ...
    {
        "image_id": xxx,
        "caption": xxx
    }
]

```
* A example can be found [here](https://github.com/OpenNLPLab/FAVDBench/releases/download/r-pt-model/prediction_coco_fmt.json).