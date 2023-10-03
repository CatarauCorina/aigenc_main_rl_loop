import numpy as np
import os
import re
import time

import torch
import itertools
from torch.utils import data
from torchvision.transforms import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AffDataset(data.Dataset):
    def __init__(self, data_dir, split='train', n_frames_per_set=5):
        self.n_frames_per_set = n_frames_per_set
        self.path = data_dir
        splits = {
            'test': slice(100),
            'valid': slice(100, 200),
            'train': slice(200, 1595)
        }
        #self.people = os.listdir(self.path)[splits[split]]
        self.ds = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "create_aff_ds\\train")
        self.object_types = os.listdir(self.ds)
        cutoff = None if split == 'train' else 1
        self.videos = [sorted(os.listdir(os.path.join(self.ds, obj)))[:cutoff]
                       for obj in self.object_types]
        self.videos = list(itertools.chain.from_iterable(self.videos))
        self.n = len(self.videos)
        self.transforms_img = transforms.Compose(
            [
                transforms.Resize(64, interpolation=Image.Resampling.BICUBIC),
                transforms.PILToTensor(),
            ]
        )

    def extract_object_info(self, object_str):
        pattern = r'^(.*?)_(\d+)\.png$'
        match = re.match(pattern, object_str)

        if match:
            object_type = match.group(1)
            object_start_pos = int(match.group(2))
            return object_type, object_start_pos
        else:
            # Return None if there is no match
            return None, None

    def __getitem__(self, item):
        #object_type = self.object_types[item]
        object_instance = self.videos[item]
        object_type, object_start_pos = self.extract_object_info(object_instance)
        video_path = os.path.join(self.ds, object_type)
        count_objects_of_type = len(os.listdir(video_path))
        if object_start_pos + self.n_frames_per_set > count_objects_of_type - 1:
            object_start_pos = count_objects_of_type - 1 - self.n_frames_per_set

        # hack to solve multiprocessing rng issue
        seed = int(str(time.time()).split('.')[1])
        np.random.seed(seed=seed)



        end = object_start_pos + self.n_frames_per_set
        frames = [f'{object_type}_{i}.png' for i in range(object_start_pos, end)]

        images = []
        for frame in frames:
            img = Image.open(os.path.join(self.ds, object_type, frame))
            img_tensor = self.transforms_img(img)
            img_tensor = torch.tensor(img_tensor, dtype=torch.float32)
            images.append(img_tensor)
        images = torch.stack(images).to(device)

        return images

    def get_object_by_type(self, object_type):
        video_path = os.path.join(self.ds, object_type)
        count_objects_of_type = len(os.listdir(video_path))
        object_start_pos = np.random.randint(1, count_objects_of_type)
        if object_start_pos + self.n_frames_per_set > count_objects_of_type - 1:
            object_start_pos = count_objects_of_type - 1 - self.n_frames_per_set

        seed = int(str(time.time()).split('.')[1])
        np.random.seed(seed=seed)

        end = object_start_pos + self.n_frames_per_set
        frames = [f'{object_type}_{i}.png' for i in range(object_start_pos, end)]

        images = []
        for frame in frames:
            img = Image.open(os.path.join(self.ds, object_type, frame))
            img_tensor = self.transforms_img(img)
            img_tensor = torch.tensor(img_tensor, dtype=torch.float32)
            images.append(img_tensor)
        images = torch.stack(images).to(device)

        return images



    def __len__(self):
        return self.n
