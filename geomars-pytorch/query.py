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
from utils import load_encoder
import math

def getClassFromPath(inPath):
    return os.path.split(inPath)[0][-3:]

def getAP(queryPath, matchesList):
    queryLabel = getClassFromPath(queryPath)
    labelList = []
    for match in matchesList:
        labelList.append(getClassFromPath(match[0]))
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

    latStr = coordComponent[:sepIndex]
    latStr = latStr[:-1], latStr[-1]
    if latStr[1] == "N":
        lat = int(latStr[0])
    else:
        lat = -int(latStr[0])

    lonStr = coordComponent[sepIndex:]
    lonStr = lonStr[:-1], lonStr[-1]
    if lonStr[1] == "E":
        lon = int(lonStr[0])
    else:
        lon = -int(lonStr[0])

    return lat, lon

def getOnMarsDistance(queryPath, matchesList):
    lat1, lon1 = getCoordinatesFromPath(queryPath)

    marsRadius = 3389.5

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    distanceList = []
    for match in matchesList:
        lat2, lon2 = getCoordinatesFromPath(match[0])
        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = marsRadius * c
        distanceList.append(distance)

    return sum(distanceList)/len(distanceList) , distanceList

def image_grid(imgs, rows, cols):

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def query(input):

    # define device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Computation device: ', device)

    # initialize the model

    model = CBIRModel(useProjector=False)
    model.to(device)

    if hp.DOMAIN_ADAPTION:
        model.useEncoder = False
        target_transform = WTransform1D(num_features=hp.DENSENET_NUM_FEATURES, group_size=hp.DA_GROUP_SIZE)

    #Load state dict
    state_dict_path = os.path.join(os.getcwd(), "outputs/model_best.pth")
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

        encoder = load_encoder()
        encoder.requires_grad_(False)
        encoder.eval()
        encoder.to(device)

    with torch.no_grad():
        image = Image.open(input)

        image = image.convert("RGB")
        image_data = data_transform(image).to(device)
        image_data = image_data.unsqueeze(0)

        if hp.DOMAIN_ADAPTION:
            output = model(target_transform(encoder(image_data).squeeze()))
        else:
            output = model(image_data)

        output = output.cpu().detach().numpy()
        hashCode = np.empty(hp.HASH_BITS).astype(np.int8)
        hashCode = ((np.sign(output -0.5)+1)/2)

        feature_dict = pickle.load(open("feature_db.p", "rb"))
        query = hashCode
        matches_list = []
        for key in feature_dict.keys():
            dist = hamming(query, np.array(feature_dict[key][0]))
            matches_list.append( (key, dist))

    matches_list.sort(key= lambda x : x[1])
    matches_list = matches_list[:64]

    print("Average Precision for this query is: ", getAP(input, matches_list))
    coordinate_distance, distance_list = getOnMarsDistance(input, matches_list)
    print("Avg. image Location distance on the surface of Mars: ", coordinate_distance)
    return matches_list, getAP(input, matches_list), coordinate_distance, distance_list

if __name__ == '__main__':
    input = sys.argv[1]
    matches_list,_,_,_ = query(input)

    image = Image.open(input)
    image.show()

    feature_dict = pickle.load(open("feature_db.p", "rb"))
    images = []
    for match in matches_list:
        image = Image.open(match[0])
        print(match[0] ,np.array(feature_dict[match[0]][0]), match[1])
        images.append(image)

    grid = image_grid(images, 8, 8)
    grid.show()
