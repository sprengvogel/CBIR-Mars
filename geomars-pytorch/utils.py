import torch
import torch.nn as nn
import os
import hparams as hp


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def load_encoder():
    if hp.DENSENET_TYPE == "imagenet":
        encoder = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)
        encoder = torch.nn.Sequential(*(list(encoder.children())[:-1]), nn.AvgPool2d(7))
    elif hp.DENSENET_TYPE == "domars16k_classifier":
        encoder = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=False)
        num_ftrs = encoder.classifier.in_features
        encoder.classifier = nn.Linear(num_ftrs, 15)
        state_dict_path = os.path.join(os.getcwd(), "models/densenet121_classifier.pth")
        encoder.load_state_dict(torch.load(state_dict_path))
        encoder = torch.nn.Sequential(*(list(encoder.children())[:-1]), nn.AvgPool2d(7))
    elif hp.DENSENET_TYPE == "domars16k_triplet":
        encoder = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=False)
        encoder = torch.nn.Sequential(*(list(encoder.children())[:-1]), nn.AvgPool2d(7))
        state_dict_path = os.path.join(os.getcwd(), "models/densenet121_triplet.pth")
        encoder.load_state_dict(torch.load(state_dict_path))
    else:
        print("Specifiy correct densenet type string in hparams.py.")
        exit(1)

    return encoder
