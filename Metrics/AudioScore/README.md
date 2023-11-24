# AudioScore

**AudioScore** assesses the accuracy of audio descriptions by computing the product of the extracted audio-visual-text unit features, where `CLIP` is used to extract features for video frames and the corresponding descriptions and `PaSST` is used for audio waves. 

```math
\begin{aligned}
\mathbf e_a=\mathrm{PaSST}(\mathbf A),\mathbf e_v=\mathrm{CLIP}(\mathbf V),  \mathbf e_t = \mathrm{CLIP} (\mathbf T), \\
s = \left(\frac{1}{2} \cos(\mathbf e_a, \mathbf e_t)  + \frac{1}{2} \cos(\mathbf e_a, \mathbf e_v) + 1 \right) \times 0.5,\\
    \mathbb{AS}(\mathbf A,\mathbf V,\mathbf T) = \mathbf f(s),
    \mathbf f(x) = a \exp (-b\exp (-c x))
\end{aligned}
```
We set c=10 empirically, and choose values for a and b ($a=\frac{1}{e^{-0.69e^{-10}}}$, b=0.693) to force specific values of x and f(x) (x=1, f(x)=1 for $a=\frac{1}{e^{-0.69e^{-10}}}$, and x=0, f(x)=0.5 for b=0.693).

## Installation
1. Activation environment
   ```bash
   conda activate FAVDBench
   ```
2. Install CLIP
    ```bash
    pip install ftfy regex tqdm
    pip install git+https://github.com/openai/CLIP.git
    ```
3. [ViT-B/32](https://github.com/openai/CLIP) is required to download under python env
    ```python
    import clip

    clip.load("ViT-B/32", jit=False)
    ```

4. To proceed, download the required pretrained [TriLip](https://github.com/OpenNLPLab/FAVDBench/releases/download/r-metric-score/TriLip.bin) model and put it into the `model` directory.
    ```bash
    Metrics/AudioScore
    |-- models  
    |   |--TriLip.bin
    ```

    |        |                                                 URL                                                 |              md5sum              |
    | :----: | :-------------------------------------------------------------------------------------------------: | :------------------------------: |
    | TriLip | 👍 [TriLip.bin](https://github.com/OpenNLPLab/FAVDBench/releases/download/r-metric-score/TriLip.bin) | 6baef8a9b383fa7c94a4c56856b0af6d |


## Score

```bash
python score.py \
    --pred_path PATH_to_PREDICTION_JSON_in_COCO_FORMAT \
```
**📝Note:** 
* `Prediction` is requried to be converted into [coco format](../README.md#coco-format).
* Default path to store results locates in `output` directory.

### Example

[result.csv](./output/default_model/result.csv) lists the sample results computed by the AudioScore.