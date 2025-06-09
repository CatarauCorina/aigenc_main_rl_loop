import numpy as np
import os
import re
import time

import torch
import itertools
from torch.utils import data
from torchvision.transforms import transforms
from PIL import Image
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AffDatasetTriplet(data.Dataset):
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
        all_videos, video_types, reversed_video_types = self.find_all_videos(self.video_classes)
        # self.videos = [sorted(os.listdir(dir)) for dir in self.folders_object_types]
        self.videos = all_videos
        self.video_types = video_types
        self.reversed_video_types = reversed_video_types
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
        reversed_video_keys = {}
        for dir in dirs:
            obj_dir = os.listdir(os.path.join(self.ds, dir))
            all_videos.extend(obj_dir)
            for obj in obj_dir:
                video_types_keys[obj] = dir
                if dir not in reversed_video_keys:
                    reversed_video_keys[dir] = [obj]
                else:
                    reversed_video_keys[dir].append(obj)
        return all_videos, video_types_keys, reversed_video_keys

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

    def create_tensor_stack(self, video_path, frames):
        images = []
        for frame in frames:
            img = Image.open(os.path.join(video_path, frame))
            img_tensor = self.transforms_img(img)
            img_tensor = torch.tensor(img_tensor, dtype=torch.float32)
            images.append(img_tensor)
        images = torch.stack(images).to(device)
        return images
    def __getitem__(self, item):
        #object_type = self.object_types[item]
        object_instance = self.videos[item]
        object_type = self.video_types[str(object_instance)]
        other_objects_same_type = self.reversed_video_types[object_type]
        pos_pair = random.choice(other_objects_same_type)
        other_objects = list(self.reversed_video_types.keys())
        other_objects.remove(object_type)
        neg_pair_type = random.choice(other_objects)
        neg_pair_instance = random.choice(self.reversed_video_types[neg_pair_type])

        video_path_1 = os.path.join(self.ds, object_type, object_instance)
        video_path_2 = os.path.join(self.ds, object_type, pos_pair)
        video_path_3 = os.path.join(self.ds, neg_pair_type, neg_pair_instance)

        count_objects_of_type = len(os.listdir(video_path_1))
        frames = os.listdir(video_path_1)
        frames2 = os.listdir(video_path_2)
        frames_neg = os.listdir(video_path_3)

        images1 = self.create_tensor_stack(video_path_1, frames)
        images2 = self.create_tensor_stack(video_path_2, frames2)
        images_neg = self.create_tensor_stack(video_path_3, frames_neg)

        return images1,images2, images_neg

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
    train_dataset = AffDatasetTriplet(data_dir=args.data_dir, split='train',
                               n_frames_per_set=5)
    # test_dataset = AffDataset(data_dir=args.data_dir, split='valid',
    #                                        n_frames_per_set=5)
    datasets = (train_dataset)
    print(len(train_dataset))

if __name__ == '__main__':
    main()