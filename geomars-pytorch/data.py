import torch
import time
from random import randrange
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision import transforms, datasets

class TripletDataset(datasets.ImageFolder):

    #def __init__(self, root, transform):
    #    super().__init__(root, transform)

    def __init__(self, root, transform):
        super().__init__(root, transform)
        #print(list(vars(self))[:-1])
        #print(self.class_to_idx)
        #print(self.classes)


        self.samples_dict = {key: [] for key in self.class_to_idx.values()}
        for sample in self.samples:
            self.samples_dict[sample[1]].append(sample[0])

        #self.triplets = []
        #for sample in self.samples:
        #    positive_class = sample[1]
        #    negative_class = randrange(0, len(self.classes))

        #    positive_sample_idx = randrange(0, len(self.samples_dict[positive_class]))
        #    negative_sample_idx = randrange(0, len(self.samples_dict[negative_class]))

        #    anchor = sample[0]
        #    positive = self.samples_dict[positive_class][positive_sample_idx]
        #    negative = self.samples_dict[negative_class][negative_sample_idx]

        #    self.triplets.append( (anchor, positive, negative) )
        #print(self.triplets)

    def __getitem__(self, index):

        sample = self.samples[index]

        positive_class = sample[1]
        negative_class = randrange(0, len(self.classes))

        positive_sample_idx = randrange(0, len(self.samples_dict[positive_class]))
        negative_sample_idx = randrange(0, len(self.samples_dict[negative_class]))

        anchor_path = sample[0]
        positive_path = self.samples_dict[positive_class][positive_sample_idx]
        negative_path = self.samples_dict[negative_class][negative_sample_idx]

        #self.triplets.append( (anchor, positive, negative) )

        #anchor_path, positive_path, negative_path = self.triplets[index]
        #print(anchor_path, positive_path, negative_path)

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
