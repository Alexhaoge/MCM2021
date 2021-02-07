from typing import Tuple
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torchvision import transforms


def get_dataset(img_list: pd.DataFrame, size: int = 256) -> TensorDataset:
    n = len(img_list)
    tran = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.2, 1)),
        transforms.ToTensor()
    ])
    X = torch.empty((n, 3, size, size))
    y = torch.Tensor(img_list['class'].values)
    for i, id in zip(img_list.index, img_list['id']):
        X[i] = tran(Image.open('data/subimg/image/'+str(id)+'.png'))
    return TensorDataset(X, y)
    

def get_dataset_aug(aug_list: pd.DataFrame, size: int = 256, times: int = 20, save: bool = False) -> TensorDataset:
    id = transforms.Lambda(lambda x: x)
    FlipAndRotate = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180, expand=True),
        transforms.RandomRotation(180, expand=False),
        transforms.RandomAffine(180, (0.3, 0.3), (0.4, 1), 3)
    ]
    Crop = [
        transforms.RandomResizedCrop(size),
        transforms.RandomCrop(size, pad_if_needed=True),
        transforms.CenterCrop(size),
        transforms.RandomPerspective(distortion_scale=0.3),
        transforms.Pad(256),
        transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomErasing(),
            transforms.ToPILImage()
        ]),
        id
    ]
    Color = [
        transforms.RandomGrayscale(0.2),
        id,
        transforms.GaussianBlur(9),
        transforms.ColorJitter(0.8, 0.8, 0.8, 0.4)
    ]
    aug_list.reset_index(drop=True, inplace=True)
    n = times * len(aug_list)
    X = torch.empty((n, 3, size, size))
    y = torch.ones((n))
    for i, id in zip(aug_list.index, aug_list['id']):
        img = Image.open('data/subimg/image/'+str(id)+'.png')
        for j in range(times):
            pipe = transforms.Compose([
                transforms.RandomChoice(FlipAndRotate),
                transforms.RandomChoice(Crop),
                transforms.RandomChoice(Color),
                transforms.Resize((size, size)),
                transforms.ToTensor()
            ])
            X[i*times+j] = pipe(img)
    if save:
        tran = transforms.ToPILImage()
        for i in range(n):
            tran(X[i]).save('data/subimg/aug/'+str(i)+'.png')           
    return TensorDataset(X, y)
            

def get_infer(infer_list: pd.DataFrame, size: int = 256) -> TensorDataset:
    n = len(infer_list)
    tran = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    X = torch.empty((n, 3, size, size))
    for i, id in zip(infer_list.index, infer_list['id']):
        X[i] = tran(Image.open('data/subimg/image/'+str(id)+'.png'))
    return TensorDataset(X)


def get_data_list(infer=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    img_list = pd.read_csv('data/subimg/list.csv')
    if infer:
        img_list = img_list[img_list['class'] == -1]
        return img_list
    img_list = img_list[img_list['class'] != -1]
    augment_list = img_list[(img_list.FileName.str.startswith('positive'))|(img_list['class']==1)]
    return img_list, augment_list


def main(batch: int = 16) -> DataLoader:
    origin, aug_list = get_data_list()
    ds = get_dataset(origin)
    aug = get_dataset_aug(aug_list)
    data_loader = DataLoader(ConcatDataset([ds, aug]), batch, True)

if __name__ == '__main__':
    data_loader = main()
    for X, y in enumerate(data_loader):
        print(X, y)
        break