

import torch
import torchvision
import torchvision.transforms as transforms

import os

DATADIR_train='./data/Videos_train_valid/train'
DATADIR_test= './data/Videos_train_valid/valid'

import os
import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision.transforms.transforms import Resize

IMG_SIZE=224
max_frames = 8
batch_size = 2
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
    frames = np.array(frames)
    frames = torch.from_numpy(frames)
    frames = torch.reshape(frames, (3, max_frames, IMG_SIZE, IMG_SIZE) )
    return np.array(frames)

def prepare_data_train(data_dir, batch_size, ):


    dataset = datasets.DatasetFolder(data_dir, loader= load_video, extensions=['mp4'])
    num_images=len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, num_images

classes=os.listdir(DATADIR_train)
classes.sort()
num_classes=len(classes)



dataloader_train, train_num_images=prepare_data_train(DATADIR_train,batch_size)
