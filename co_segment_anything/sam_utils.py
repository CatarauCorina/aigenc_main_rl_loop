import torch
import torchvision.models as models
import torchvision.transforms as transforms
import os
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import numpy as np, cv2
import torch
import scipy
from PIL import Image
import torch.nn.functional as F
from memory_graph.init_concept_space import ObjectConcept

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn
from torchvision import models

#original_model = models.alexnet(pretrained=True)
resnet = models.resnet50(pretrained=True)


class SegmentAnythingObjectExtractor(object):

    def __init__(self, checkpoint="sam_vit_b_01ec64.pth", model_type="vit_b", no_objects=6):
        self.checkpoint_path = "../co_segment_anything/checkpoints"
        print(os.getcwd())
        self.checkpoint_file = f'{self.checkpoint_path}/{checkpoint}'
        self.sam_model = sam_model_registry[model_type](checkpoint=self.checkpoint_file).to(device).eval()
        self.sam_encoder = SamPredictor(self.sam_model)
        self.no_objects = no_objects
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam_model,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.97,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Requires open-cv to run post-processin
        )
        self.transform = transforms.ToTensor()
        self.pil_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(128, interpolation=Image.Resampling.BICUBIC),
            transforms.ToTensor()])
        self.im_height = 64
        self.im_width = 64
        # Remove the classification layer (the last layer)
        self.resnet_encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.resnet_encoder = self.resnet_encoder.to(device).eval()

        self.resnet_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


        self.linear_layer_resnet = torch.nn.Linear(2048, 512).to(device)
        self.linear_layer_squeeze = torch.nn.Linear(1000, 16).to(device)

    def pass_through_resnet(self, obj_tensor):
        features = self.resnet_encoder(obj_tensor)
        # Reduce the dimensionality to 512
        reduced_features = torch.nn.functional.relu(features)  # Apply ReLU activation
        reduced_features = torch.nn.functional.adaptive_avg_pool2d(reduced_features, (1, 1))
        reduced_features = self.linear_layer_resnet(reduced_features.view(-1, 2048))
        return reduced_features

    def extract_objects(self, frame, use_model='resnet'):
        objects = []
        objects_resnet = []

        #frame_reduced = self.pil_transform(frame).permute(1, 2, 0).detach().numpy()
        frame_reduced = cv2.resize(frame, dsize=(self.im_height, self.im_width), interpolation=cv2.INTER_CUBIC)
        with torch.no_grad():
            masks = self.mask_generator.generate(frame_reduced)
        masks = masks[:self.no_objects]
        for mask in masks:
            mask_inverted = np.invert(mask['segmentation']).astype(int)
            mask_arr = np.stack((mask_inverted,) * 3, axis=-1)
            masked = np.where(mask_arr == 0, frame_reduced, 0)
            if use_model == 'resnet':
                tensor = self.resnet_transform(masked).float()
            objects.append(tensor)

        obj_tensor = torch.stack(objects).to(device)
        if use_model == 'resnet':
            with torch.no_grad():
                reduced_features = self.pass_through_resnet(obj_tensor)
                tensor_img_reduced = self.resnet_transform(frame_reduced).float().to(device)
                encoded_state = self.pass_through_resnet(tensor_img_reduced.unsqueeze(0))

        encoded_objs = reduced_features

        if len(masks) < self.no_objects:
            encoded_objs = F.pad(input=encoded_objs, pad=(0, 0, self.no_objects-len(masks), 0), mode='constant', value=0)

        return encoded_objs.unsqueeze(0).unsqueeze(0).to(device), encoded_state

    def find_objects_positions(self, frame):
        objects_positions = []
        frame_reduced = cv2.resize(frame, dsize=(self.im_height, self.im_width), interpolation=cv2.INTER_CUBIC)
        masks = self.mask_generator.generate(frame_reduced)
        masks = masks[:self.no_objects]
        for mask in masks:
            mask_inverted = np.invert(mask['segmentation']).astype(int)
            mask_center = self.find_mask_center(mask_inverted)
            objects_positions.append(mask_center)
        return objects_positions


    def find_mask_center(self, mask):
        indices = np.where(mask == 0)
        x_mean = np.mean(indices[1])
        y_mean = np.mean(indices[0])
        return (x_mean, y_mean)


