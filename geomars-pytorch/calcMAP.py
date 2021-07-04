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
import pickle
import math
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

def getAP(queryLabel, labelList):
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


def  calc_map():

    # define device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Computation device: ', device)

    # initialize the model
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

    ctx_test = datasets.ImageFolder(root="./data/test", transform=data_transform)
    db_loader = torch.utils.data.DataLoader(
        ctx_test,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model.eval()
    feature_dict = pickle.load(open("feature_db.p", "rb"))
    #print(model)
    mAP = 0
    marsDistanceAvg = 0
    with torch.no_grad():
        for bi, data in tqdm(enumerate(db_loader), total=int(len(ctx_test))):
            image_data,image_label = data
            image_data = image_data.to(device)
            image_label = image_label.cpu().detach().numpy()[0]
            output = model(image_data)

            output = output.cpu().detach().numpy()
            hashCode = np.empty(hp.HASH_BITS).astype(np.int8)
            hashCode = ((np.sign(output -0.5)+1)/2)

            query = hashCode
            matches_list = []
            label_list = []
            sample_fname, _ = db_loader.dataset.samples[bi]
            for i, key in enumerate(feature_dict.keys()):
                label = feature_dict[key][1]
                dist = hamming(query, np.array(feature_dict[key][0]))
                matches_list.append( (key, dist))
                label_list.append(label)

            #Sort matches by distance and sort labels in the same way
            matches_list, label_list = (list(t) for t in zip(*sorted(zip(matches_list,label_list), key= lambda x : x[0][1])))
            matches_list = matches_list[:64]
            label_list = label_list[:64]
            average_precision = getAP(image_label, label_list)
            mAP += average_precision
            marsDistanceAvg += getOnMarsDistance(sample_fname, matches_list)[0]
    mAP /= int(len(ctx_test))
    marsDistanceAvg /= int(len(ctx_test))
    print(mAP)
    print(marsDistanceAvg)
    return mAP, marsDistanceAvg

if __name__ == '__main__':
    calc_map()
