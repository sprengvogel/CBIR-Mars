import torch
import time
from random import randrange, choice, seed
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision import transforms, datasets
from sklearn.cluster import KMeans
import numpy as np
import cv2

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

    def __init__(self, root, transform, interclasstriplets=False, n_clusters = None):
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
                data = np.array([cv2.imread(path, cv2.IMREAD_GRAYSCALE).flatten() for path in data_paths])

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
            self.n_clusters = 30
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


class TripletDataset(datasets.ImageFolder):

    def __init__(self, root, transform, test_mode = False):
        super().__init__(root, transform)
        #print(list(vars(self))[:-1])
        #print(self.class_to_idx)
        #print(self.classes)

        if test_mode:
            seed(42)

        print("Generating dictionary to assign classes to img paths.")

        self.samples_dict = {key: [] for key in self.class_to_idx.values()}
        for sample in self.samples:
            self.samples_dict[sample[1]].append(sample[0])

    def __getitem__(self, index):

        sample = self.samples[index]

        positive_class = sample[1]
        negative_class = getRandomNumber(0, len(self.classes), positive_class)#randrange(0, len(self.classes))

        sample_idx = self.samples_dict[positive_class].index(sample[0])
        positive_sample_idx = getRandomNumber(0, len(self.samples_dict[positive_class]), sample_idx)#randrange(0, len(self.samples_dict[positive_class]))
        negative_sample_idx = getRandomNumber(0, len(self.samples_dict[negative_class]))#randrange(0, len(self.samples_dict[negative_class]))

        #print("positive class: ", sample[1])
        #print("negative class: ", negative_class)
        #print("sample   index: ", sample_idx)
        #print("p sample index: ", positive_sample_idx)
        anchor_path = sample[0]
        positive_path = self.samples_dict[positive_class][positive_sample_idx]
        negative_path = self.samples_dict[negative_class][negative_sample_idx]

        anchor = self.loader(anchor_path)
        positive = self.loader(positive_path)
        negative = self.loader(negative_path)
        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return (anchor_path, anchor), (positive_path, positive), (negative_path, negative)

    def __len__(self):
        return len(self.samples)

class InterClassTripletDataset(TripletDataset):

    def __init__(self, root, transform, test_mode = False):
        super().__init__(root, transform, test_mode)
        #print(list(vars(self))[:-1])
        #print(self.class_to_idx)
        #print(self.classes)
        if test_mode:
            seed(42)

        #self.samples_dict = {key: [] for key in self.class_to_idx.values()}
        #for sample in self.samples:
        #    self.samples_dict[sample[1]].append(sample[0])

        print("Load all images as numpy arrays and calculate mean vector and covariance matrix per class.")

        self.class_statistics = {key: [] for key in self.class_to_idx.values()}
        self.data = {key: [] for key in self.class_to_idx.values()}
        self.standardized_data = {key: [] for key in self.class_to_idx.values()}
        for key in tqdm(self.class_to_idx.values(), total=len(self.class_to_idx)):
            data_paths = self.samples_dict[key]
            data = np.array([cv2.imread(path, cv2.IMREAD_GRAYSCALE).flatten() for path in data_paths])

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
        self.n_clusters = 30
        kmeans = KMeans(n_clusters=self.n_clusters , random_state=0).fit(self.imgs)
        self.kmeans_labels = kmeans.labels_

        self.clusters_dict = {}
        for cluster_idx in range(self.n_clusters):
            self.clusters_dict[cluster_idx] = {}

        for idx, label in enumerate(kmeans.labels_):
            sample_path, sample_class = self.samples[idx]
            if sample_class not in self.clusters_dict[label]:
                self.clusters_dict[label][sample_class] = []
            self.clusters_dict[label][sample_class].append( sample_path)

        print("Generation of inter-class feature groups completed.")

    def __getitem__(self, index):

        (anchor_path, anchor), (positive_path, positive), (negative_path, negative) = super().__getitem__(index)

        sample_class = self.samples[index][1]
        sample_cluster = self.kmeans_labels[index]

        ic_positive_cluster = self.clusters_dict[sample_cluster]
        ic_negative_cluster = self.clusters_dict[getRandomNumber(0, self.n_clusters, sample_cluster)]


        ic_positives_idx_list = list(ic_positive_cluster.keys())
        if len(ic_positives_idx_list) > 1:
            ic_positives_idx_list.remove(sample_class)

        ic_positive_class = ic_positive_cluster[ choice(ic_positives_idx_list) ]

        ic_negatives_idx_list = list(ic_negative_cluster.keys())
        if sample_class in ic_negatives_idx_list  and len(ic_negatives_idx_list) > 1:
            ic_negatives_idx_list.remove(sample_class)
            if ic_positive_class in ic_negatives_idx_list and len(ic_negatives_idx_list) > 1:
                ic_negatives_idx_list.remove(ic_positive_class)

        ic_negative_class = ic_negative_cluster[ choice(ic_negatives_idx_list) ]

        ic_positive_path = ic_positive_class[getRandomNumber(0, len(ic_positive_class))]
        ic_negative_path = ic_negative_class[getRandomNumber(0, len(ic_negative_class)) ]


        ic_positive = self.loader(ic_positive_path)
        ic_negative = self.loader(ic_negative_path)
        if self.transform is not None:
            ic_positive = self.transform(ic_positive)
            ic_negative = self.transform(ic_negative)

        return (anchor_path, anchor), (positive_path, positive), (negative_path, negative), (ic_positive_path, ic_positive), (ic_negative_path, ic_negative_path)
