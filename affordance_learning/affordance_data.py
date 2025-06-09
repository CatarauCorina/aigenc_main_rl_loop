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
        self.ds = os.path.join(os.path.dirname(os.getcwd()), "create_aff_ds\\objects_obs_grouped")
        self.object_types = os.listdir(self.ds)
        cutoff = None if split == 'train' else 1
        # self.folders_object_types = [sorted(os.listdir(os.path.join(self.ds, obj)))[:cutoff]
        #                for obj in self.object_types]
        self.video_classes = [obj for obj in self.object_types]
        self.folder_paths = [os.path.join(self.ds, obj) for obj in self.object_types]
        all_videos, video_types = self.find_all_videos(self.video_classes)
        # self.videos = [sorted(os.listdir(dir)) for dir in self.folders_object_types]
        self.videos = all_videos
        self.video_types = video_types
        self.n = len(self.videos)
        self.transforms_img = transforms.Compose(
            [
                transforms.Resize(64, interpolation=Image.Resampling.BICUBIC),
                transforms.PILToTensor(),
            ]
        )

    def find_all_videos(self, dirs):
        all_videos = []
        video_types_keys = {}
        for dir in dirs:
            obj_dir = os.listdir(os.path.join(self.ds, dir))
            all_videos.extend(obj_dir)
            for obj in obj_dir:
                video_types_keys[obj] = dir
        return all_videos, video_types_keys

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
        object_type = self.video_types[str(object_instance)]
        video_path = os.path.join(self.ds, object_type, object_instance)
        count_objects_of_type = len(os.listdir(video_path))
        frames = os.listdir(video_path)

        images = []
        for frame in frames:
            img = Image.open(os.path.join(video_path, frame))
            img_tensor = self.transforms_img(img)
            img_tensor = torch.tensor(img_tensor, dtype=torch.float32)
            images.append(img_tensor)
        images = torch.stack(images).to(device)
        images_classif = images.permute(1, 0, 2, 3)  # Change shape to (nr_channels, sequence_length, height, width)
        images_classif = images_classif.to(device)

        return images,images_classif, self.video_classes.index(object_type)

    def get_object_by_type(self, object_type):
        video_path = os.path.join(self.ds, object_type)
        count_objects_of_type = len(os.listdir(video_path))
        object_of_type = os.listdir(video_path)
        import random
        object_random = random.sample(object_of_type, 1)

        frames = os.listdir(os.path.join(video_path, object_random[0]))

        images = []
        for frame in frames:
            img = Image.open(os.path.join(video_path, object_random[0], frame))
            img_tensor = self.transforms_img(img)
            img_tensor = torch.tensor(img_tensor, dtype=torch.float32)
            images.append(img_tensor)
        images = torch.stack(images).to(device)

        return images, object_random[0]



    def __len__(self):
        return self.n


import argparse
def main():
    parser = argparse.ArgumentParser(description='Neural Statistician Aff Experiment')

    # required
    parser.add_argument('--data-dir', type=str, default='create_aff_ds',
                        help='location of formatted Omniglot data')
    args = parser.parse_args()
    train_dataset = AffDataset(data_dir=args.data_dir, split='train',
                               n_frames_per_set=5)
    # test_dataset = AffDataset(data_dir=args.data_dir, split='valid',
    #                                        n_frames_per_set=5)
    datasets = (train_dataset)
    print(train_dataset[0])

if __name__ == '__main__':
    main()