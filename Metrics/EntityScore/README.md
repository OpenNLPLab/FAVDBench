# EntityScore

**EntityScore** measures the extent to which consistently referred words or series of words, known as entities and often manifested as nouns, in the predicted text match those in the annotated text. We use an off-the-shelf Natural Language Toolkit library to extract nouns as entities. The mathematical expression of the EntityScore $\mathbb{ES}$ is as follows:

```math
\begin{aligned}
R(\mathbf p,\mathbf r)=\frac{\#\{ \mathbf p \cap \mathbf r \}}{\#\{\mathbf r\}}, \\
C(\mathbf p,\mathbf r)=\frac{\cos(\mathrm{T5}(\mathbf p), \mathrm{T5}(\mathbf r))+1} 2, \\
\mathbb{ES}(\mathbf p,\mathbf r)= \frac{2 R(\mathbf p,\mathbf r) C(\mathbf p, \mathbf r)}{R(\mathbf p,\mathbf r) + C(\mathbf p, \mathbf r)}
\end{aligned}
```

## Installation
1. Activation environment
   ```bash
   conda activate FAVDBench
   ```
2. Install NLTK toolkit
    ```bash
    pip install --user -U nltk
    ```
3. [T5-base](https://huggingface.co/t5-base) is required to download under python env
    ```python
    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5EncoderModel.from_pretrained('t5-base')
    ```

## Score

```bash
python score.py \
    --pred_path PATH_to_PREDICTION_JSON_in_COCO_FORMAT \
    --refe_path AVLFormer/datasets/metadata/test.caption_coco_format.json \
    --model_path t5-base
```
**📝Note:** 
* `Prediction` is requried to be converted into [coco format](../README.md#coco-format).

### Example

```python
prediction = "on stage a man speaks. on stage a man speaking had a white head a black suit a white shirt and a red tie. behind the man is a white wall. in front of the man is a black microphone. the man speaks with a."

reference = "inside a man speaks into a microphone. a speech is being delivered by a man with white and black hair on his temples a black suit a white shirt and a tie with light and blue and white diagonal stripes. the man is backed by a sizable red wall. the man is in front of two microphones. he has a rhythm to his speech."

entity_score = compute_entity_score(prediction, reference, "test",
                                    tokenizer, model)

print(f"Entity Score: {entity_score:.3f}")
```