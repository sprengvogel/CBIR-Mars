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
import math

def getClassFromPath(inPath):
    return os.path.split(inPath)[0][-3:]

def getAP(queryPath, matchesList):
    queryLabel = getClassFromPath(queryPath)
    labelList = []
    for match in matchesList:
        labelList.append(getClassFromPath(match[0]))
    print(labelList)
    acc = np.zeros((0,)).astype(float)
    correct = 1
    for (i, label) in enumerate(labelList):
        if label == queryLabel:
            precision = (correct / float(i+1))
            acc = np.append(acc, [precision, ], axis=0)
            correct += 1
    if correct == 1:
        return 0.
    num = np.sum(acc)
    den = correct - 1
    return num/den

def getCoordinatesFromPath(inPath):
    pathComponents = inPath.split("_")
    coordComponent = pathComponents[-3]
    sepIndex = max(coordComponent.find("S"), coordComponent.find("N")) +1
    lonStr = coordComponent[:sepIndex]
    lonStr = lonStr[:-1], lonStr[-1]
    if lonStr[1] == "N":
        lon = int(lonStr[0])
    else:
        lon = -int(lonStr[0])
    latStr = coordComponent[sepIndex:]
    latStr = latStr[:-1], latStr[-1]
    if latStr[1] == "E":
        lat = int(latStr[0])
    else:
        lat = -int(latStr[0])

    return lon ,lat

def getOnMarsDistance(queryPath, matchesList):
    lon1, lat1 = getCoordinatesFromPath(queryPath)

    marsRadius = 3389.5

    lon1 = math.radians(lon1)
    lat1 = math.radians(lat1)
    print(lon1, lat1)
    distanceList = []
    for match in matchesList:
        lon2, lat2 = getCoordinatesFromPath(match[0])
        lon2 = math.radians(lon2)
        lat2 = math.radians(lat2)
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = marsRadius * c
        print(distance)
        distanceList.append(distance)
    print(sum(distanceList)/len(distanceList))
    return 0

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def query():

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
        if hp.USE_IMAGENET:
            encoder = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)
            encoder = torch.nn.Sequential(*(list(encoder.children())[:-1]), nn.AvgPool2d(7))
        else:
            encoder = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=False)
            num_ftrs = encoder.classifier.in_features
            encoder.classifier = nn.Linear(num_ftrs, 15)
            state_dict_path = os.path.join(os.getcwd(), "outputs/densenet121_pytorch_adapted.pth")
            encoder.load_state_dict(torch.load(state_dict_path))
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

    print("Average Precision for this query is: ", getAP(sys.argv[1], matches_list[:64]))
    #coordinate_distance = getOnMarsDistance(sys.argv[1], matches_list[:64])
    #print("Image Location distance on the surface of Mars: ", coordinate_distance)

    grid = image_grid(images, 8, 8)
    grid.show()


    #print(feature_dict)

if __name__=='main':
    query()