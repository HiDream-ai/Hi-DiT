import io
import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image


def load_h5_file(hf, path):
    # Helper function to load files from h5 file
    if path.endswith('.png'):
        rtn = np.array(PIL.Image.open(io.BytesIO(np.array(hf[path]))))
        rtn = rtn.reshape(*rtn.shape[:2], -1).transpose(2, 0, 1)
    elif path.endswith('.json'):
        rtn = json.loads(np.array(hf[path]).tobytes().decode('utf-8'))
    elif path.endswith('.npy'):
        rtn= np.array(hf[path])
    else:
        raise ValueError('Unknown file type: {}'.format(path))
    return rtn


class CustomINH5Dataset(Dataset):
    def __init__(self, data_dir):
        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}

        self.data_dir = data_dir
        self.h5_path = os.path.join(self.data_dir, "images.h5")
        self.h5_json_path = os.path.join(self.data_dir, "images_h5.json")
        self.h5f = h5py.File(self.h5_path, 'r')

        with open(self.h5_json_path, 'r') as f:
            self.h5_json = json.load(f)
        self.filelist = {fname for fname in self.h5_json}
        self.filelist = sorted(fname for fname in self.filelist if self._file_ext(fname) in supported_ext)

        labels = load_h5_file(self.h5f, "dataset.json")["labels"]
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self.filelist]
        labels = np.array(labels)
        self.labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])

    def __len__(self):
        return len(self.filelist)

    def _file_ext(self, fname):
        return os.path.splitext(fname)[1].lower()

    def __del__(self):
        self.h5f.close()

    def __getitem__(self, index):
        """
        Images should be '.png'
        """
        image_fname = self.filelist[index]
        image = load_h5_file(self.h5f, image_fname)
        return torch.from_numpy(image), torch.tensor(self.labels[index])


class CustomH5Dataset(Dataset):
    def __init__(self, data_dir, vae_latents_name="repae-invae-400k"):
        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}

        self.images_h5 = h5py.File(os.path.join(data_dir, 'images.h5'), "r")
        self.features_h5 = h5py.File(os.path.join(data_dir, f'{vae_latents_name}.h5'), "r")
        images_json = os.path.join(data_dir, 'images_h5.json')
        features_json = os.path.join(data_dir, f'{vae_latents_name}_h5.json')

        with open(images_json, 'r') as f:
            images_json = json.load(f)
        with open(features_json, 'r') as f:
            features_json = json.load(f)

        # images
        self._image_fnames = {fname for fname in images_json}
        self.image_fnames = sorted(fname for fname in self._image_fnames if self._file_ext(fname) in supported_ext)

        # features
        self._feature_fnames = {fname for fname in features_json}
        self.feature_fnames = sorted(fname for fname in self._feature_fnames if self._file_ext(fname) in supported_ext)
        
        # labels
        fname = 'dataset.json'
        labels = load_h5_file(self.features_h5, fname)['labels']
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self.feature_fnames]
        labels = np.array(labels)
        self.labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])

    def _file_ext(self, fname):
        return os.path.splitext(fname)[1].lower()

    def __len__(self):
        assert len(self.image_fnames) == len(self.feature_fnames), \
            "Number of feature files and label files should be same"
        return len(self.feature_fnames)

    def __del__(self):
        self.images_h5.close()
        self.features_h5.close()

    def __getitem__(self, idx):
        image_fname = self.image_fnames[idx]
        feature_fname = self.feature_fnames[idx]
        image_ext = self._file_ext(image_fname)

        image = load_h5_file(self.images_h5, image_fname)
        if image_ext == '.npy':
            # npy needs some extra care
            image = image.reshape(-1, *image.shape[-2:])

        features = load_h5_file(self.features_h5, feature_fname)
        return torch.from_numpy(image), torch.from_numpy(features), torch.tensor(self.labels[idx])
