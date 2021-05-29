import torch
import time
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision import transforms, datasets
import os
from PIL import Image
from data import TripletDataset

if __name__ == '__main__':

    #Change current working directory to source file location
    #os.chdir(os.path.dirname(__file__))

    batch_size = 1
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

    """ state_dict = torch.load(state_dict_path)
    new_state_dict = {}
    for key in state_dict:
        new_state_dict[key.removeprefix("net.")] = state_dict[key] """

    model.load_state_dict(torch.load(state_dict_path))
    #torch.save(model.state_dict(), "densenet121_pytorch_adapted.pth")

    data_transform = transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    ctx_test = TripletDataset(root="./data/test", transform=data_transform)
    test_loader = torch.utils.data.DataLoader(
        ctx_test, batch_size=batch_size, shuffle=False, num_workers=4
    )

    #tf_last_layer_chopped = nn.Sequential(*list(model.children())[:-1])

    #model = nn.Sequential(
    #    model,
    #    nn.AvgPool2d(7)
    #)

    criterion = nn.TripletMarginLoss()
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for bi, data in tqdm(enumerate(test_loader), total=int(len(ctx_test) / test_loader.batch_size)):
            anchor = data[0].to(device)
            positive = data[1].to(device)
            negative = data[2].to(device)

            output_anchor = model(anchor)
            output_pos = model(positive)
            output_neg = model(negative)
            loss = criterion(output_anchor, output_pos, output_neg)
            #print(output_anchor, output_pos, output_neg)
            # add loss of each item (total items in a batch = batch size)
            #print(running_loss)
            running_loss += loss.item()
    final_loss = running_loss / len(ctx_test)
    print(f"Test Loss: {final_loss}")
