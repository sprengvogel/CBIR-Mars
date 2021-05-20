import torch
import time
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision import transforms, datasets
import os
from PIL import Image
import sys
import numpy as np
import json


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

if __name__ == '__main__':

    #Change current working directory to source file location
    os.chdir(os.path.dirname(__file__))

    batch_size = 16
    num_classes = 15
    # define device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Computation device: ', device)

    # initialize the model
    model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=False)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    model.to(device)

    #Load state dict
    state_dict_path = os.path.join(os.getcwd(), "densenet121_pytorch_adapted.pth")

    """ state_dict = torch.load(state_dict_path)
    new_state_dict = {}
    for key in state_dict:
        new_state_dict[key.removeprefix("net.")] = state_dict[key] """

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



    tf_last_layer_chopped = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    #print(model)
    with torch.no_grad():
        image = Image.open(sys.argv[1])
        image = image.convert("RGB")
        print(np.array(image).shape)
        image_data = data_transform(image).to(device)
        image_data = image_data.unsqueeze(0)
        output = tf_last_layer_chopped(image_data)
        pool = torch.nn.AvgPool2d(7)
        fVector = pool(output).squeeze().cpu().numpy()
        print(fVector)
        image.show()


        db_file = open("feature_db.json", "r")
        feature_dict = json.load(db_file)
        query = fVector
        matches_list = []
        for key in feature_dict.keys():
            #print(feature_dict[key])
            #print(query)
            dist = np.linalg.norm(query - np.array(feature_dict[key]))
            #if dist < best_match[1]:
            matches_list.append( (key, dist))
            print(dist)

    matches_list.sort(key= lambda x : x[1])
    images = []
    for match in matches_list[:25]:
        image = Image.open(match[0])
        #image.show()
        images.append(image)
    grid = image_grid(images, 5, 5)
    grid.show()


    #print(feature_dict)
