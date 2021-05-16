import torch
import time
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision import transforms, datasets
import os
from PIL import Image

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
    state_dict_path = os.path.join(os.getcwd(), "outputs", "model_best.pth")
    model.load_state_dict(torch.load(state_dict_path))

    rootPath = os.path.join(os.getcwd(),"data","test","aec")
    imageNames = os.listdir(rootPath)

    data_transform = transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    ctx_train = datasets.ImageFolder(root="./data/train", transform=data_transform)
    train_loader = torch.utils.data.DataLoader(
        ctx_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    ctx_val = datasets.ImageFolder(root="./data/val", transform=data_transform)
    val_loader = torch.utils.data.DataLoader(
        ctx_val, batch_size=batch_size, shuffle=True, num_workers=8
    )

    tf_last_layer_chopped = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    print(model)
    with torch.no_grad():
        for bi, data in tqdm(enumerate(train_loader), total=int(len(ctx_train) / train_loader.batch_size)):
            image_data = (data[0].to(device))
            outputs = tf_last_layer_chopped(image_data)
            for output in outputs:
                pool = torch.nn.AvgPool2d(7)
                fVector = pool(output).numpy()
                #print(pool(output).shape)
        for bi, data in tqdm(enumerate(val_loader), total=int(len(ctx_val) / val_loader.batch_size)):
            image_data = (data[0].to(device))
            outputs = tf_last_layer_chopped(image_data)
            for output in outputs:
                pool = torch.nn.AvgPool2d(7)
                fVector = pool(output).numpy()
                #print(pool(output).shape)