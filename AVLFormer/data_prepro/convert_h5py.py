import glob
import os
import sys

import h5py
import numpy as np

prefix = sys.argv[1]

files = sorted(glob.glob(f'datasets/audio_mp3/{prefix}/*.mp3'))

# print(files)

available_size = len(files)
print(available_size)

dt = h5py.vlen_dtype(np.dtype('uint8'))
ds = h5py.special_dtype(vlen=str)

os.makedirs('datasets/audio_hdf', exist_ok=True)
with h5py.File('datasets/audio_hdf' + prefix + '_mp3.hdf', 'w') as hf:
    audio_name = hf.create_dataset(
        'audio_name', shape=((available_size,)), dtype=ds)
    waveform = hf.create_dataset('mp3', shape=((available_size,)), dtype=dt)
    for i, file in enumerate(files):
        if i % 1000 == 1:
            print(f'{i}/{available_size}')
            print(a.shape)

        a = np.fromfile(file, dtype='uint8')
        audio_name[i] = os.path.basename(file)[:-4]
        waveform[i] = a

print(np.fromfile(file, dtype='uint8').shape)
print('Done!', prefix)
