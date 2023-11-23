import argparse
import os
import random
from shutil import copy2

import clip
from dataset import TriLipDataset
from misc import MyLogger
from model.trilip import TriLip
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm


def scale(x):
    x = (x / 100 - 0.5) * 2
    x = np.exp(
        -0.693179 * np.exp(-10 * x)) * (1 / np.exp(-0.693179 * np.exp(-10)))
    return x * 100


@torch.no_grad()
def inference(samples, ck_path):

    device = 'cuda:0'
    batch_size = 128

    # seed
    seed = 2048
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # io
    _id = os.path.basename(samples)[:-5]
    os.makedirs(f'output/{_id}', exist_ok=True)
    logger = MyLogger(path=f'output/{_id}/log.log', name=__name__)
    copy2(samples, f'output/{_id}/{os.path.basename(samples)}')

    # model
    vl_model, transform = clip.load("ViT-B/32", device=device, jit=False)
    model = TriLip(vl_model)
    ck = torch.load(ck_path, map_location='cpu')['model']
    _ck = dict()
    for k, v in ck.items():
        _ck[k[7:]] = v
    ck_result = model.load_state_dict(_ck)
    logger.info(ck_result)

    model.cuda(device)
    model = torch.nn.DataParallel(model, device_ids=[0])

    # dataset
    dataset_val = TriLipDataset('../../AVLFormer/datasets/metadata/test.caption.tsv',
                                transform,
                                training=False,
                                inference=True,
                                inference_meta=samples)
    sampler_val = None

    dataloader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True,
    )

    # eval mode
    model.eval()
    case_id, vals = [], []
    for _, batch_data in tqdm(enumerate(dataloader_val)):
        img, txt, wav, name = batch_data

        with torch.no_grad():
            with autocast(enabled=True):
                logits, _ = model(img, txt, wav)
                logits = torch.diagonal(logits, 0).cpu().detach().numpy()
                logits = scale(logits)

        for n, v in zip(name, logits):
            case_id.append(n)
            vals.append(v)
            logger.info(f'{n}: {v}')

    _result = np.mean(vals)
    case_id.append('mean')
    vals.append(_result)
    logger.info(f'mean: {_result}')

    pd.DataFrame({
        'case': case_id,
        'score': vals
    }).to_csv(f'output/{_id}/result.csv', index=False, lineterminator='\n')


if __name__ == '__main__':
    DEFAULT_MODEL = 'model/TriLip.bin'

    parser = argparse.ArgumentParser()
    # path configs
    parser.add_argument("--pred_path",
                        "-p",
                        default="path/to/prediction.json",
                        type=str,
                        required=False,
                        help="predction path")
    args = parser.parse_args()

    inference(samples=args.pred_path,
              ck_path=DEFAULT_MODEL)
