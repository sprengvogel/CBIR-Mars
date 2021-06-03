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

if __name__ == '__main__':

    # define device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Computation device: ', device)

    # initialize the model
    densenet = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)
    densenet.to(device)
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
    with torch.no_grad():
        for bi, data in tqdm(enumerate(db_loader), total=int(len(ctx_test))):
            image_data,image_label = data
            image_data = image_data.to(device)
            image_label = image_label.cpu().detach().numpy()[0]
            image_data = densenet(image_data)
            output = model(image_data)

            output = output.cpu().detach().numpy()
            hashCode = np.empty(hp.HASH_BITS).astype(np.int8)
            hashCode = ((np.sign(output -0.5)+1)/2)
            
            query = hashCode
            matches_list = []
            label_list = []
            sample_fname, _ = db_loader.dataset.samples[bi]
            for key in feature_dict.keys():
                label = feature_dict[key][1]
                dist = hamming(query, np.array(feature_dict[key][0]))
                matches_list.append( (key, dist))
                label_list.append(label)
            #Sort matches by distance and sort labels in the same way
            matches_list, label_list = (list(t) for t in zip(*sorted(zip(matches_list,label_list), key= lambda x : x[0][1])))
            average_precision = getAP(image_label, label_list)
            mAP += average_precision
    mAP /= int(len(ctx_test))
    print(mAP)
