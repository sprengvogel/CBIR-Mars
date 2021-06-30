import torch
import time
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision import transforms, datasets
import torch.nn.functional as F
import os
from PIL import Image
import sys
import numpy as np
import pickle
import hparams as hp
from CBIRModel import CBIRModel
from scipy.spatial.distance import hamming
from whitening import WTransform1D


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

if __name__ == '__main__':

    # define device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Computation device: ', device)

    # initialize the model
    model = CBIRModel()
    model.to(device)

    if hp.DOMAIN_ADAPTION:
        model.useEncoder = False
        target_transform = WTransform1D(num_features=hp.DENSENET_NUM_FEATURES, group_size=hp.DA_GROUP_SIZE)

    #Load state dict
    state_dict_path = os.path.join(os.getcwd(), "outputs/model_last.pth")
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(state_dict_path))
        if hp.DOMAIN_ADAPTION:
            target_transform.load_state_dict(torch.load(os.path.join(os.getcwd(), 'outputs/target_transform.pth')))
    else:
        model.load_state_dict(torch.load(state_dict_path, map_location=torch.device('cpu')))
        if hp.DOMAIN_ADAPTION:
            target_transform.load_state_dict(torch.load(os.path.join(os.getcwd(), 'outputs/target_transform.pth'), map_location=torch.device('cpu')))

    data_transform = transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )


    model.eval()

    if hp.DOMAIN_ADAPTION:
        target_transform.eval()
        encoder = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)
        encoder = torch.nn.Sequential(*(list(encoder.children())[:-1]), nn.AvgPool2d(7))
        encoder.requires_grad_(False)
        encoder.eval()
        encoder.to(device)

    with torch.no_grad():
        image = Image.open(sys.argv[1])
        image = image.convert("RGB")
        #print(np.array(image).shape)
        image_data = data_transform(image).to(device)
        image_data = image_data.unsqueeze(0)

        if hp.DOMAIN_ADAPTION:
            print(encoder(image_data).squeeze())
            print(target_transform(encoder(image_data).squeeze()))
            output = model(target_transform(encoder(image_data).squeeze()))
        else:
            output = model(image_data)

        output = output.cpu().detach().numpy()
        hashCode = np.empty(hp.HASH_BITS).astype(np.int8)
        hashCode = ((np.sign(output -0.5)+1)/2)

        image.show()


        feature_dict = pickle.load(open("feature_db.p", "rb"))
        query = hashCode
        print(hashCode)
        matches_list = []
        for key in feature_dict.keys():
            #print(np.array(feature_dict[key]))
            dist = hamming(query, np.array(feature_dict[key][0]))
            matches_list.append( (key, dist))
            #print(dist)

    matches_list.sort(key= lambda x : x[1])
    images = []
    for match in matches_list[:64]:
        image = Image.open(match[0])
        print(match[0] ,np.array(feature_dict[match[0]][0]), match[1])
        images.append(image)
    grid = image_grid(images, 8, 8)
    grid.show()


    #print(feature_dict)
