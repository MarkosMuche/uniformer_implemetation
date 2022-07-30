

import os
import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision.transforms.transforms import Resize

IMG_SIZE=128
max_frames = 20
import cv2
import numpy as np


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

def load_video(path, max_frames=max_frames, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f'bad video file{path}')
                cap.release
                return None
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

def prepare_data_train(data_dir, input_size):

    train_transforms = transforms.Compose([transforms.Resize(300),
                                        transforms.CenterCrop(input_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    dataset = datasets.DatasetFolder(data_dir, loader= load_video, extensions=['mp4'])
    num_images=len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader, num_images


# def prepare_data_test(data_dir,input_size):
#     test_transforms = transforms.Compose([transforms.Resize(300),
#                                       transforms.CenterCrop(input_size),
#                                       transforms.ToTensor(),
#                                       transforms.Normalize([0.485, 0.456, 0.406],
#                                                            [0.229, 0.224, 0.225])])
#     dataset = datasets.ImageFolder(data_dir, transform=test_transforms)
#     num_images=len(dataset)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
#     return dataloader, num_images
