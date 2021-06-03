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
import json
import hparams as hp
from CBIRModel import CBIRModel
from scipy.spatial.distance import hamming


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
    densenet = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)
    densenet.to(device)
    densenet.requires_grad_(False)
    densenet.eval()

    model = CBIRModel()
    model.to(device)

    #Load state dict
    state_dict_path = os.path.join(os.getcwd(), "outputs/model_best.pth")

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(state_dict_path))
    else:
        model.load_state_dict(torch.load(state_dict_path, map_location=torch.device('cpu')))
    #torch.save(model.state_dict(), "densenet121_pytorch_adapted.pth")

    data_transform = transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )


    model.eval()
    #print(model)
    with torch.no_grad():
        image = Image.open(sys.argv[1])
        image = image.convert("RGB")
        #print(np.array(image).shape)
        image_data = data_transform(image).to(device)
        image_data = image_data.unsqueeze(0)

        dense_image_data = densenet(image_data)
        output = model(dense_image_data)

        output = output.cpu().detach().numpy()
        hashCode = np.empty(hp.HASH_BITS).astype(np.int8)
        hashCode = ((np.sign(output -0.5)+1)/2)

        image.show()


        with open("feature_db.json", "r") as db_file:
            feature_dict = json.load(db_file)
        query = hashCode
        #print(hashCode)
        matches_list = []
        for key in feature_dict.keys():
            #print(np.array(feature_dict[key]))
            dist = hamming(query, np.array(feature_dict[key]))
            matches_list.append( (key, dist))
            #print(dist)

    matches_list.sort(key= lambda x : x[1])
    images = []
    for match in matches_list[:64]:
        image = Image.open(match[0])
        images.append(image)
    grid = image_grid(images, 8, 8)
    grid.show()


    #print(feature_dict)
