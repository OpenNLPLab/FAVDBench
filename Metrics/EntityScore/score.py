import argparse
import json
import warnings

import nltk
from nltk.stem.porter import PorterStemmer
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

warnings.filterwarnings('ignore')
from transformers import logging

logging.set_verbosity_error()

porter_stemmer = PorterStemmer()


def load_json_cc_fmt(file_path):
    fmt = dict()
    with open(file_path, 'r') as f:
        jobj = json.load(f)

    for itm in jobj:
        fmt[itm['image_id']] = itm['caption']

    return fmt


def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent


def noun_extracter(sentence):
    output = preprocess(sentence)
    Nouns = []
    for rec in output:
        if "NN" in rec[1]:
            Nouns.append(rec[0])

    nns = list(set([porter_stemmer.stem(i) for i in list(set(Nouns))]))
    return nns


def entity_avg(pred, ref):
    if len(ref) == 0:
        return 0

    count = sum(1 for n in pred if n in ref)

    return count / len(ref)


def semantic_avg(pred, ref, tokenizer, model):
    pred_ids = tokenizer(pred,
                         return_tensors="pt",
                         padding=True,
                         truncation=True)
    refe_ids = tokenizer(ref,
                         return_tensors="pt",
                         padding=True,
                         truncation=True)

    max_length = max(pred_ids["input_ids"].shape[1],
                     refe_ids["input_ids"].shape[1])

    pred_ids = pad_input_ids(pred_ids, max_length)
    refe_ids = pad_input_ids(refe_ids, max_length)

    pred_output = model.encoder(
        input_ids=pred_ids["input_ids"],
        attention_mask=pred_ids["attention_mask"]).last_hidden_state
    refe_output = model.encoder(
        input_ids=refe_ids["input_ids"],
        attention_mask=refe_ids["attention_mask"]).last_hidden_state

    avg_per_sentence = compute_cosine_similarity(pred_output, refe_output)

    return avg_per_sentence


def pad_input_ids(ids_dict, max_length):
    """pad ids to maximum length"""
    for key in ids_dict:
        pad_size = max_length - ids_dict[key].shape[1]
        ids_dict[key] = torch.cat([
            ids_dict[key],
            torch.zeros(ids_dict[key].shape[0], pad_size).long()
        ],
                                  dim=1)
    return ids_dict


def compute_cosine_similarity(pred_output, refe_output):
    similarity_scores = []
    for pred in pred_output:
        scores = []
        for refe in refe_output:
            score = (F.cosine_similarity(pred, refe).mean().item() + 1) / 2
            scores.append(score)
        avg_score = min(max(scores), 1.0)
        similarity_scores.append(avg_score)

    return sum(similarity_scores) / len(similarity_scores)


def compute_entity_score(prediction, reference, key, tokenizer, model):
    prediction_nouns = noun_extracter(prediction)
    reference_nouns = noun_extracter(reference)

    if len(prediction_nouns) == 0 or len(reference_nouns) == 0:
        print(f"No nouns extracted for key: {key}")
        return -1

    comprehensiveness = semantic_avg(prediction_nouns, reference_nouns,
                                     tokenizer, model)
    precision = entity_avg(prediction_nouns, reference_nouns)

    # Prevent division by zero
    precision += 1e-4
    entity_score = 2 * comprehensiveness * precision / (comprehensiveness +
                                                        precision)

    return entity_score


def process_file(prediction_path, reference_path, tokenizer, model):
    prediction_dict = load_json_cc_fmt(prediction_path)
    reference_dict = load_json_cc_fmt(reference_path)

    entity_scores = []

    for key, prediction in tqdm(prediction_dict.items()):
        reference = reference_dict[key]

        entity_score = compute_entity_score(prediction, reference, key,
                                            tokenizer, model)

        if entity_score != -1:
            entity_scores.append(entity_score)

    avg = sum(entity_scores) / len(entity_scores)
    min_score = min(entity_scores)
    max_score = max(entity_scores)

    print(f"Average: {avg:.2f}, Range: ({min_score:.2f} - {max_score:.2f})")


def test_entity_score(model_dir):
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5EncoderModel.from_pretrained(model_dir)

    prediction = "on stage a man speaks. on stage a man speaking had a white head a black suit a white shirt and a red tie. behind the man is a white wall. in front of the man is a black microphone. the man speaks with a."
    reference = "inside a man speaks into a microphone. a speech is being delivered by a man with white and black hair on his temples a black suit a white shirt and a tie with light and blue and white diagonal stripes. the man is backed by a sizable red wall. the man is in front of two microphones. he has a rhythm to his speech."

    entity_score = compute_entity_score(prediction, reference, "test",
                                        tokenizer, model)
    print(f"Entity Score: {entity_score:.3f}")


def main():

    parser = argparse.ArgumentParser()
    # path configs
    parser.add_argument("--pred_path",
                        "-p",
                        default="path/to/prediction.json",
                        type=str,
                        required=False,
                        help="predction path")

    parser.add_argument("--refe_path",
                        "-r",
                        default="path/to/reference.json",
                        type=str,
                        required=False,
                        help="reference path")

    parser.add_argument("--model_path",
                        "-m",
                        default='t5-base',
                        type=str,
                        required=False,
                        help="load model path")

    args = parser.parse_args()

    tokenizer = T5Tokenizer.from_pretrained(args.model_path)
    model = T5EncoderModel.from_pretrained(args.model_path)

    # test_entity_score(args.model_path)
    process_file(args.pred_path, args.refe_path, tokenizer, model)


if __name__ == "__main__":
    main()
