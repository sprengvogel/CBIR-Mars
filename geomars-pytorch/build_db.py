import torch
import time
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional  as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision import transforms, datasets
import os
from PIL import Image
import pickle
from CBIRModel import CBIRModel
import hparams as hp
import numpy as np
from whitening import WTransform1D


def build_db():
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
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )


    ctx_train = datasets.ImageFolder(root="./data/database", transform=data_transform)

    db_loader = torch.utils.data.DataLoader(
        ctx_train,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    feature_dict = {}
    model.eval()

    if hp.DOMAIN_ADAPTION:
        target_transform.eval()
        if hp.DENSENET_TYPE == "imagenet":
            encoder = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)
            encoder = torch.nn.Sequential(*(list(encoder.children())[:-1]), nn.AvgPool2d(7))
        elif hp.DENSENET_TYPE == "domars16k_classifier":
            encoder = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=False)
            num_ftrs = encoder.classifier.in_features
            encoder.classifier = nn.Linear(num_ftrs, 15)
            state_dict_path = os.path.join(os.getcwd(), "outputs/densenet121_pytorch_adapted.pth")
            encoder.load_state_dict(torch.load(state_dict_path))
            encoder = torch.nn.Sequential(*(list(encoder.children())[:-1]), nn.AvgPool2d(7))
        else:
            print("Specifiy correct densenet type string in hparams.py.")
            exit(1)

        encoder.requires_grad_(False)
        encoder.eval()
        encoder.to(device)

    with torch.no_grad():
        for bi, data in tqdm(enumerate(db_loader), total=int(len(ctx_train))):# / db_loader.batch_size)):
            image_data,image_label = data
            image_label = image_label.cpu().detach().numpy()[0]
            image_data = image_data.to(device)
            if hp.DOMAIN_ADAPTION:
                #print(encoder(image_data).squeeze())
                #print(target_transform(encoder(image_data).squeeze()))
                output = model(target_transform(encoder(image_data).squeeze()))
            else:
                output = model(image_data)
            output = output.cpu().detach().numpy()

            hashCode = np.empty(hp.HASH_BITS).astype(np.int8)
            hashCode = ((np.sign(output -0.5)+1)/2)
            #print(output)
            #print(hashCode)
            sample_fname, _ = db_loader.dataset.samples[bi]
            feature_dict[sample_fname] = (hashCode.tolist(), image_label)
    #print(feature_dict)
    pickle.dump(feature_dict, open("feature_db.p", "wb"))

if __name__ == '__main__':
    build_db()
