from __future__ import print_function
import pandas as pd
import numpy as np
import glob
import os
import os.path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib.pyplot import imread
import torch.utils.data as data
from PIL import Image

csv_path = 'D:/Code/resnet/data/archive/caltech256.csv'
root = 'D:/Code/resnet/data/archive/256_ObjectCategories'


class Caltech256(data.Dataset):

    def __init__(self, root,
                 transform=None,
                 target_transform=None,
                 splits=[1, 2, 3, 4, 5],
                 csv_path=csv_path):
        # Initate some variables
        self.root = os.path.expanduser(root)
        self.splits = splits
        self.csv_path = csv_path
        self.transform = transform
        self.target_transform = target_transform

        # if absent, create csv with data split
        if not os.path.isfile(self.csv_path):
            self.create_csv()
        self.df = pd.read_csv(self.csv_path)

        # Select desired splits and re-index df
        self.df = self.df[self.df.fold.isin(splits)]
        self.df = self.df.reset_index(drop=True)

        # Get class names and weights
        self.classNames = set([p.split('/')[-2] for p in self.df.path])
        self.classNames = sorted(self.classNames)
        self.nclasses = len(self.classNames)

        # Get class weighting
        unique, counts = np.unique(self.df.label, return_counts=True)
        weights = sorted(dict(zip(unique, counts)).items())
        weights = np.array(weights)[:, 1].astype(float)
        weights = min(weights) / weights
        self.weights = torch.from_numpy(weights).float()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        imgPath, target = self.df['path'][index], self.df['label'][index]
        imgPath = os.path.join(self.root, imgPath)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(imgPath)
        if img.mode != 'RGB':
            img = img.convert(mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.df)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        tmp2 = self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp))
        fmt_str += '{0}{1}\n'.format(tmp, tmp2)
        tmp = '    Target Transforms (if any): '
        tmp2 = self.target_transform.__repr__().replace('\n',
                                                        '\n' + ' ' * len(tmp))
        fmt_str += '{0}{1}'.format(tmp, tmp2)
        return fmt_str

    def create_csv(self):
        ''' Create a csv file with matching image path, label and fold
        '''
        dataset_path = self.root

        labels = []
        paths = []
        folds = []

        for root, _, files in os.walk(dataset_path):
            root = root.split('/')[-1]
            root = root + '/'

            for idx, f in enumerate(files):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    label = int(root.split('.')[0]) - 1
                    path = root + f
                    fold = int(idx / float(len(files)) * 5) + 1
                    labels.append(label)
                    paths.append(path)
                    folds.append(fold)

        df = pd.DataFrame({'path': paths, 'label': labels, 'fold': folds})
        df.to_csv(self.csv_path)


if __name__ == "__main__":
    cal_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    trainset = Caltech256(root=root,
                          transform=cal_transform,
                          splits=[1, 2, 3, 4])
    testset = Caltech256(root=root,
                         transform=cal_transform,
                         splits=[5])
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                               shuffle=True, num_workers=2)

    for images, labels in train_loader:
        print(images.shape)  # 输出：(batch_size, channels, height, width)
        print(labels.shape)  # 输出：(batch_size,)

