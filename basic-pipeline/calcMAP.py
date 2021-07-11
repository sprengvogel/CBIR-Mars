
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

SELF_TRAINED = True

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def getClassFromPath(path):
    return path

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

def  calc_map(path, classifier=False):


    # define device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Computation device: ', device)

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

    model = torch.nn.Sequential(*(list(model.children())[:-1]), nn.AvgPool2d(7))
    model.to(device)
    model.eval()

    data_transform = transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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

    feature_dict = {}

    db_file = open("feature_db.json", "r")
    feature_dict = json.load(db_file)
    #print(model)
    mAP = 0
    with torch.no_grad():
        for bi, data in tqdm(enumerate(db_loader), total=int(len(ctx_test))):
            image_data, image_label = data
            image_data = image_data.to(device)
            image_label = image_label.cpu().detach().numpy()[0]
            output = model(image_data)
            fVector = output.cpu().numpy()
            fVector = fVector.squeeze()

            query = fVector
            matches_list = []
            label_list = []
            sample_fname, _ = db_loader.dataset.samples[bi]

            for key in feature_dict.keys():
                #print(feature_dict[key])
                #print(query)
                label = feature_dict[key][1]
                dist = np.linalg.norm(query - np.array(feature_dict[key][0]))
                #if dist < best_match[1]:
                matches_list.append( (key, dist))
                label_list.append(label)
            #Sort matches by distance and sort labels in the same way
            matches_list, label_list = (list(t) for t in zip(*sorted(zip(matches_list,label_list), key= lambda x : x[0][1])))
            matches_list = matches_list[:64]
            label_list = label_list[:64]
            average_precision = getAP(image_label, label_list)
            mAP += average_precision
    mAP /= int(len(ctx_test))
    print(mAP)
    return mAP

if __name__ == '__main__':
    path = "outputs/model_best.pth"
    calc_map(path)
