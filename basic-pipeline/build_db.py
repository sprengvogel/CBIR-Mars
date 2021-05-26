import torch
import time
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision import transforms, datasets
import os
from PIL import Image
import json


if __name__ == '__main__':

    #Change current working directory to source file location
    os.chdir(os.path.dirname(__file__))

    #batch_size = 16
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
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(state_dict_path))
    else:
        model.load_state_dict(torch.load(state_dict_path, map_location=torch.device('cpu')))


    data_transform = transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    ctx_train = datasets.ImageFolder(root="./data/train", transform=data_transform)
    db_loader = torch.utils.data.DataLoader(
        ctx_train,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    feature_dict = {}

    tf_last_layer_chopped = nn.Sequential(*list(model.children())[:-1])
    pool = torch.nn.AvgPool2d(7)
    model.eval()

    with torch.no_grad():
        for bi, data in tqdm(enumerate(db_loader), total=int(len(ctx_train))):# / db_loader.batch_size)):
            #if bi > 10:
            #    break
            image_data = (data[0].to(device))
            output = tf_last_layer_chopped(image_data)

            fVector = pool(output).cpu().numpy()
            fVector = fVector.squeeze()
            sample_fname, _ = db_loader.dataset.samples[bi]
            feature_dict[sample_fname] = fVector.tolist()

    db_file = open("feature_db.json", "w")
    db_file.write(json.dumps(feature_dict))
    db_file.close()
