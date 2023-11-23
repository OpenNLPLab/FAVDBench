import io
import json

from PIL import Image
import av
import clip
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor


class TriLipDataset(Dataset):

    def __init__(self,
                 meta,
                 transform,
                 training,
                 inference=False,
                 inference_meta=None) -> None:
        super().__init__()

        data = pd.read_csv(meta, sep='\t', header=None)
        self.metas = data[0].to_numpy()
        if not inference:
            self.caps = data[1].to_numpy()
        else:
            assert inference_meta is not None
            self.caps = dict()
            with open(inference_meta, 'r') as f:
                self.caps = json.load(f)
            #     jobj = json.load(f)
            # for obj in jobj:
            #     self.caps[obj['image_id']] = obj['caption']

        with open('../../AVLFormer/datasets/audio_hdf/train_mp3.hdf', 'rb') as f:
            self.audios_train = h5py.File(io.BytesIO(f.read()), 'r')
        with open('../../AVLFormer/datasets/audio_hdf/val_mp3.hdf', 'rb') as f:
            self.audios_val = h5py.File(io.BytesIO(f.read()), 'r')
        with open('../../AVLFormer/datasets/audio_hdf/test_mp3.hdf', 'rb') as f:
            self.audios_test = h5py.File(io.BytesIO(f.read()), 'r')

        self.sample_rate = 32000
        self.clip_length = 10 * self.sample_rate

        self.training = training
        self.inference = inference
        # self.img_transform = self._img_transform()
        self.img_transform = transform

    def __getitem__(self, idx):

        name = self.metas[idx].split('/')
        if self.training:
            if name[0] == 'train':
                audio_file = self.audios_train
                img_dir = '../../AVLFormer/datasets/frames/train-32frames'
                a_idx = idx
            else:
                audio_file = self.audios_val
                img_dir = '../../AVLFormer/datasets/frames/val-32frames'
                a_idx = idx - 7500
        else:
            audio_file = self.audios_test
            img_dir = '../../AVLFormer/datasets/frames/test-32frames'
            a_idx = idx

        # audio_name = audio_file['audio_name'][idx].decode()
        wave_form = self.decode_mp3(audio_file['mp3'][a_idx])
        wave_form = self.pydub_augment(waveform=wave_form)
        wave_form = self.pad_or_truncate(x=wave_form,
                                         audio_length=self.clip_length)

        if self.training:
            wave_form = self.roll_func(x=wave_form.reshape(1, -1))
        else:
            wave_form = wave_form.reshape(1, -1)

        base_name = name[1][:-4]
        img = Image.open(f'{img_dir}/{base_name}_frame0001.jpg')
        img = self.img_transform(img)

        if self.inference:
            txt = self.caps[base_name].split('. ')[-1]
            # txt = '. '.join(self.caps[base_name].split('. ')[-2:])
        else:
            txt = json.loads(self.caps[idx])[0]['caption'].split('. ')[-1]
        txt = self.txt_transform(txt)

        return img, txt, wave_form, base_name

    def __len__(self):
        return len(self.metas)

    def _img_transform(self):
        return Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073),
                      (0.26862954, 0.26130258, 0.27577711)),
        ])

    def txt_transform(self, x):
        return clip.tokenize(x, truncate=True).squeeze()

    def decode_mp3(self, mp3_arr):
        """
        decodes an array if uint8 representing an mp3 file
        :rtype: np.array
        """
        container = av.open(io.BytesIO(mp3_arr.tobytes()))
        stream = next(s for s in container.streams if s.type == 'audio')
        a = []
        for _, packet in enumerate(container.demux(stream)):
            for frame in packet.decode():
                a.append(frame.to_ndarray().reshape(-1))
        waveform = np.concatenate(a)
        if waveform.dtype != 'float32':
            raise RuntimeError('Unexpected wave type')
        return waveform

    def pydub_augment(self, waveform, gain_augment=7, ir_augment=0):
        if gain_augment:
            gain = torch.randint(gain_augment * 2, (1, )).item() - gain_augment
            amp = 10**(gain / 20)
            waveform = waveform * amp
        return waveform

    def pad_or_truncate(self, x, audio_length):
        """Pad all audio to specific length."""
        if len(x) <= audio_length:
            return np.concatenate(
                (x, np.zeros(audio_length - len(x), dtype=np.float32)), axis=0)
        else:
            return x[0:audio_length]

    def roll_func(self, x, axis=1, shift=None, shift_range=50):
        x = torch.as_tensor(x)
        sf = shift
        if shift is None:
            sf = int(np.random.randint(-shift_range, shift_range))

        return x.roll(sf, axis)
