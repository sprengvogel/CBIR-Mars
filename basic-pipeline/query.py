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

def getClassFromPath(inPath):
    return os.path.split(inPath)[0][-3:]

def getAP(queryPath, matchesList):
    queryLabel = getClassFromPath(queryPath)
    labelList = []
    for match in matchesList:
        labelList.append(getClassFromPath(match[0]))
    #print(labelList)
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

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def query(modelpath, imagepath, classifier=False):

    batch_size = 16

    # define device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Computation device: ', device)

    if modelpath != None:
        #Select state dict
        state_dict_path = os.path.join(os.getcwd(), path)


        # initialize the model
        model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=False)

        if classifier == True:
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, 15)

        if torch.cuda.is_available():
            model.load_state_dict(torch.load(state_dict_path))
        else:
            model.load_state_dict(torch.load(state_dict_path, map_location=torch.device('cpu')))
    else:
        # initialize the model
        model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)

    model = torch.nn.Sequential(*(list(model.children())[:-1]), nn.AvgPool2d(7))
    model.to(device)

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
        image = Image.open(imagepath)
        image = image.convert("RGB")
        image_data = data_transform(image).to(device)
        image_data = image_data.unsqueeze(0)
        output = model(image_data)
        fVector = output.cpu().numpy()
        fVector = fVector.squeeze()

        db_file = open("feature_db.json", "r")
        feature_dict = json.load(db_file)
        query = fVector
        matches_list = []
        for key in feature_dict.keys():
            dist = np.linalg.norm(query - np.array(feature_dict[key][0]))
            matches_list.append((key, dist))

    matches_list.sort(key= lambda x : x[1])
    matches_list = matches_list[:64]

    print("Average Precision for this query is: ", getAP(imagepath, matches_list))
    return matches_list, getAP(imagepath, matches_list)

if __name__ == '__main__':
    path = "outputs/model_best.pth"
    matches_list,_ = query(path, sys.argv[1], classifier=True)


    image = Image.open(sys.argv[1])
    image.show()

    db_file = open("feature_db.json", "r")
    feature_dict = json.load(db_file)
    images = []
    for match in matches_list:
        image = Image.open(match[0])
        images.append(image)

    grid = image_grid(images, 8, 8)
    grid.show()
