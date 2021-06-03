from CBIRModel import CBIRModel
import torch
import time
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision import transforms, datasets
import os
from PIL import Image
from data import TripletDataset
import hparams as hp
from CBIRModel import CBIRModel
from loss import criterion

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
    model.load_state_dict(torch.load(state_dict_path))

    data_transform = transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    ctx_test = TripletDataset(root="./data/test", transform=data_transform)
    test_loader = torch.utils.data.DataLoader(
        ctx_test, batch_size=hp.BATCH_SIZE, shuffle=False, num_workers=4
    )

    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for bi, data in tqdm(enumerate(test_loader), total=int(len(ctx_test) / test_loader.batch_size)):
            print(data)
            anchor = data[0].to(device)
            positive = data[1].to(device)
            negative = data[2].to(device)

            dense_anchor = densenet(anchor)
            dense_positive = densenet(positive)
            dense_negative = densenet(negative)
            
            output_anchor = model(dense_anchor)
            output_pos = model(dense_positive)
            output_neg = model(dense_negative)

            loss = criterion(output_anchor, output_pos, output_neg)
            running_loss += loss.item()
    final_loss = running_loss / len(ctx_test)
    print(f"Test Loss: {final_loss}")
