from numpy import mod
import torch
import time
from random import randrange
from torch.nn.modules.container import ModuleList
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision import transforms, datasets
from data import TripletDataset, InterClassTripletDataset, ImageFolderWithLabel
from CBIRModel import CBIRModel
import hparams as hp
from loss import criterion as hashing_criterion
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from pytorch_metric_learning import losses, miners, distances, reducers, testers

# train the model
def train(model, dataloader, train_dict):
    model.train()
    running_loss = 0.0
    for bi, data in tqdm(enumerate(dataloader), total=int(len(ctx_train) / dataloader.batch_size)):
        inputs = torch.stack([train_dict[x] for x in data[0]])
        labels = data[1]

        # zero grad the optimizer
        optimizer.zero_grad()
        embeddings = model(inputs)

        triplet_indices_tuple = triplet_mining(embeddings, labels)
        triplet_loss = criterion(embeddings, labels, triplet_indices_tuple)
        #print(triplet_indices_tuple)
        #print(embeddings)
        hashing_loss = hashing_criterion(embeddings)
        #print("triplet loss: ", triplet_loss)
        #print("combined hash loss: ", hashing_loss)
        loss = triplet_loss + hashing_loss

        '''if hp.INTERCLASSTRIPLETS == True:
            ic_positive = torch.stack([train_dict[x] for x in data[3][0]])
            ic_negative = torch.stack([train_dict[x] for x in data[4][0]])
            ic_output_pos = model(ic_positive)
            ic_output_neg = model(ic_negative)

            loss += criterion(output_anchor, ic_output_pos, ic_output_neg)'''
        # backpropagation
        loss.backward()
        #plot_grad_flow(model.named_parameters())
        # update the parameters
        optimizer.step()
        # add loss of each item (total items in a batch = batch size)
        running_loss += loss.item()
    final_loss = running_loss / len(ctx_train)

    return final_loss

#validate model
def validate(model, dataloader, val_dict, epoch):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for bi, data in tqdm(enumerate(dataloader), total=int(len(ctx_val) / dataloader.batch_size)):
            inputs = torch.stack([val_dict[x] for x in data[0]])
            labels = data[1]

            embeddings = model(inputs)

            triplet_indices_tuple = triplet_mining(embeddings, labels)
            loss = criterion(embeddings, labels, triplet_indices_tuple) + hashing_criterion(embeddings)

            # add loss of each item (total items in a batch = batch size)
            running_loss += loss.item()
    final_loss = running_loss / len(ctx_val)

    return final_loss

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.cpu().detach().abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)

if __name__ == '__main__':

    data_transform = transforms.Compose(
        [
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )



    ctx_train = ImageFolderWithLabel(root="./data/train", transform=data_transform)
    train_loader = torch.utils.data.DataLoader(
        ctx_train,
        batch_size=hp.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    ctx_val = ImageFolderWithLabel(root="./data/val", transform=data_transform)
    val_loader = torch.utils.data.DataLoader(
        ctx_val,
        batch_size=hp.BATCH_SIZE,
        shuffle=True,
        num_workers=8
    )

    ctx_test = ImageFolderWithLabel(root="./data/test", transform=data_transform)
    test_loader = torch.utils.data.DataLoader(
        ctx_test,
        batch_size=hp.BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    # define device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Computation device: ', device)

    #Use densenet on every image
    ctx_train_densenet = datasets.ImageFolder(root="./data/train", transform=data_transform)
    train_loader_densenet = torch.utils.data.DataLoader(
        ctx_train_densenet,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    ctx_val_densenet = datasets.ImageFolder(root="./data/val", transform=data_transform)
    val_loader_densenet = torch.utils.data.DataLoader(
        ctx_val_densenet,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    encoder = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)
    encoder = torch.nn.Sequential(*(list(encoder.children())[:-1]), nn.AvgPool2d(7))
    encoder.requires_grad_(False)
    encoder.eval()
    encoder.to(device)
    train_file_path = Path("./train.p")
    if train_file_path.is_file():
        train_dict = pickle.load(open("train.p","rb"))
    else:
        train_dict = {}
        for bi, data in tqdm(enumerate(train_loader_densenet), total=int(len(ctx_train_densenet) / train_loader_densenet.batch_size)):
            image_data,image_label = data
            sample_fname, _ = train_loader_densenet.dataset.samples[bi]
            image_data = image_data.to(device)
            output = encoder(image_data).squeeze().detach().clone()
            train_dict[sample_fname] = output
        pickle.dump(train_dict, open("train.p", "wb"))

    val_file_path = Path("./val.p")
    if val_file_path.is_file():
        val_dict = pickle.load(open("val.p","rb"))
    else:
        val_dict = {}
        for bi, data in tqdm(enumerate(val_loader_densenet), total=int(len(ctx_val_densenet) / val_loader_densenet.batch_size)):
            image_data,image_label = data
            sample_fname, _ = val_loader_densenet.dataset.samples[bi]
            image_data = image_data.to(device)
            output = encoder(image_data).squeeze().detach().clone()
            val_dict[sample_fname] = output
        pickle.dump(val_dict, open("val.p", "wb"))

    # initialize the model
    model = CBIRModel(useEncoder=False)
    model.to(device)

    # define optimizer. Also initialize learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=hp.LR, betas=hp.ADAM_BETAS)

    distance = distances.LpDistance()#distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low = 0)#DoNothingReducer()#
    criterion = losses.TripletMarginLoss(margin = 0.2, distance = distance, reducer = reducer)
    triplet_mining = miners.TripletMarginMiner(margin = 0.2, distance = distance, type_of_triplets = "hard")

    train_loss, val_loss = [], []
    start = time.time()
    best_loss = 1000
    best_epoch = 0

    # start training and validating
    for epoch in range(hp.EPOCHS):
        print(f"Epoch {epoch + 1} of {hp.EPOCHS}")
        train_epoch_loss = train(model, train_loader, train_dict)
        val_epoch_loss = validate(model, val_loader, val_dict, epoch)
        # save model with best loss
        if val_epoch_loss < best_loss:
            best_epoch = epoch
            best_loss = val_epoch_loss
            print("Saved Model. Best Epoch: " + str(best_epoch+1))
            torch.save(model.state_dict(), 'outputs/model_best.pth')
        print(f"Train Loss: {train_epoch_loss}")
        print(f"Val Loss: {val_epoch_loss}")
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
    end = time.time()
    #plt.show()
    print(f"Finished training in: {((end - start) / 60):.3f} minutes")
    print(f"Best performing Epoch:: {best_epoch}.")
