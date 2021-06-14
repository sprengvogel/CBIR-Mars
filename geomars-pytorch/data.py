import torch
import time
from random import randrange
import random
from scipy import ndimage
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision import transforms, datasets
from skimage import feature, filters
import numpy as np
import cv2
from PIL import Image


class MultiviewDataset(datasets.ImageFolder):
    def __init__(self, root, transform):
        super().__init__(root, transform)

        self.samples_dict = {key: [] for key in self.class_to_idx.values()}
        for sample in self.samples:
            self.samples_dict[sample[1]].append(sample[0])

    def add_noise(self, image):
        # https://www.geeksforgeeks.org/add-a-salt-and-pepper-noise-to-an-image-with-python/

        # Getting the dimensions of the image
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        row, col, _ = img.shape

        # Randomly pick some pixels in the image for coloring them white
        number_of_pixels = random.randint(300, 1000)
        for i in range(number_of_pixels):
            y_coord = random.randint(0, row - 1)
            x_coord = random.randint(0, col - 1)

            img[y_coord][x_coord][0] = 255
            img[y_coord][x_coord][1] = 255
            img[y_coord][x_coord][2] = 255

        # Randomly pick some pixels in the image for coloring them black
        number_of_pixels = random.randint(300, 1000)
        for i in range(number_of_pixels):
            y_coord = random.randint(0, row - 1)
            x_coord = random.randint(0, col - 1)

            img[y_coord][x_coord][0] = 0
            img[y_coord][x_coord][1] = 0
            img[y_coord][x_coord][2] = 0

        return img

    def lbp(self, img, eps=1e-7):
        numPoints = 24
        radius = 8
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lbp = feature.local_binary_pattern(img, numPoints, radius, method='uniform')
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, numPoints + 3),
                                 range=(0, numPoints + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        # return the histogram of Local Binary Patterns
        return hist


    def colorHist(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

        return hist

    def __getitem__(self, index):

        sample = self.samples[index]

        image = self.loader(sample[0])

        resize_transform = transforms.Resize([224, 224])
        image = resize_transform(image)

        img_noise = self.add_noise(image)

        # fd, view1 = feature.hog(img_noise, orientations=5, pixels_per_cell=(25, 25),
        #                             cells_per_block=(1, 1), visualize=True)

        view1 = cv2.Canny(img_noise,100,200)
        view1 = cv2.cvtColor(view1,cv2.COLOR_GRAY2RGB)
        view1_PIL = Image.fromarray(view1)

        # view2 = self.lbp(img_noise)

        view3 = ndimage.sobel(cv2.cvtColor(img_noise, cv2.COLOR_BGR2GRAY))
        view3 = cv2.cvtColor(view3, cv2.COLOR_GRAY2RGB)
        view3_PIL = Image.fromarray(view3)

        if self.transform is not None:
            view1_PIL = self.transform(view1_PIL)
            view3_PIL = self.transform(view3_PIL)

        # cv2.imshow("sobel", view3)
        # cv2.waitKey(0)
        #view4 = self.colorHist(img_noise)

        return view1_PIL, view3_PIL, sample[1]

    def __len__(self):
        return len(self.samples)




class TripletDataset(datasets.ImageFolder):

    def __init__(self, root, transform):
        super().__init__(root, transform)

        self.samples_dict = {key: [] for key in self.class_to_idx.values()}
        for sample in self.samples:
            self.samples_dict[sample[1]].append(sample[0])
        t = 0


    def __getitem__(self, index):

        sample = self.samples[index]

        positive_class = sample[1]
        negative_class = randrange(0, len(self.classes))

        positive_sample_idx = randrange(0, len(self.samples_dict[positive_class]))
        negative_sample_idx = randrange(0, len(self.samples_dict[negative_class]))

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

        return anchor, positive, negative

    def __len__(self):
        return len(self.samples)
