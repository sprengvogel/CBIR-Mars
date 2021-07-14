import torch
import time
from random import randrange, choice, seed
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision import transforms, datasets
from sklearn.cluster import KMeans #MiniBatchKMeans as
import numpy as np
import cv2



class MultiviewDataset(datasets.ImageFolder):

    def __init__(self, root, transform):
        super().__init__(root, transform)
        self.transform = transform
        self.samples_dict = {key: [] for key in self.class_to_idx.values()}
        for sample in self.samples:
            self.samples_dict[sample[1]].append(sample[0])

    def __getitem__(self, index):
        sample = self.samples[index]
        image = self.loader(sample[0])

        data_transform = transforms.Compose(
        [
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        )
        if self.transform is not None:
            view_1 = data_transform(image)
            view_2 = self.transform(image)

        return view_1, view_2

def removeclassdoublings(indices_tuple, labels):

    matches1 = (labels[indices_tuple[0]] == labels[indices_tuple[1]])
    matches2 = (labels[indices_tuple[0]] == labels[indices_tuple[2]])
    matches = ~(matches1 | matches2)

    return ( indices_tuple[0][matches], indices_tuple[1][matches], indices_tuple[2][matches])

def getRandomNumber(start, end, exclude=None):
    number_range = list(range(start, end))
    if exclude is not None:
        number_range.remove(exclude)
    return choice(number_range)

class ImageFolderWithLabel(datasets.ImageFolder):

    def __init__(self, root, transform, interclasstriplets=False, n_clusters = None, features_dict = None):
        super().__init__(root, transform)
        self.interclasstriplets = interclasstriplets
        self.n_clusters = n_clusters

        if self.interclasstriplets:
            print("Generating dictionary to assign classes to img paths.")

            self.samples_dict = {key: [] for key in self.class_to_idx.values()}
            for sample in self.samples:
                self.samples_dict[sample[1]].append(sample[0])

            print("Load all images as numpy arrays and calculate mean vector and covariance matrix per class.")

            self.class_statistics = {key: [] for key in self.class_to_idx.values()}
            self.data = {key: [] for key in self.class_to_idx.values()}
            self.standardized_data = {key: [] for key in self.class_to_idx.values()}
            for key in tqdm(self.class_to_idx.values(), total=len(self.class_to_idx)):
                data_paths = self.samples_dict[key]
                #data = np.array([cv2.imread(path, cv2.IMREAD_GRAYSCALE).flatten() for path in data_paths])
                data = np.array([features_dict[path].detach().cpu().numpy() for path in data_paths])

                self.data[key] = data
                class_mean = np.mean(data, axis=0)
                class_var = np.var(data, axis=0)
                self.class_statistics[key] = (class_mean, class_var)

                standardized_imgs = []
                for img in data:
                    standardized_img = (img - class_mean)/class_var
                    standardized_imgs.append(standardized_img)
                self.standardized_data[key] = np.array(standardized_imgs)


            print("Load images as numpy arrays into memory.")
            self.imgs = np.concatenate([self.data[key] for key in self.data.keys()], axis=0)

            print("Performing kmeans clustering ...")
            kmeans = KMeans(n_clusters=self.n_clusters , random_state=0).fit(self.imgs)
            self.kmeans_labels = kmeans.labels_

            print("Generation of inter-class feature groups completed.")

    def __getitem__(self, index):

        sample = self.samples[index]
        sample_path = sample[0]
        sample_label = sample[1]

        if self.interclasstriplets:
            interclass_label = self.kmeans_labels[index]
            return (sample_path, sample_label, interclass_label)

        return (sample_path, sample_label)

    def __len__(self):
        return len(self.samples)
