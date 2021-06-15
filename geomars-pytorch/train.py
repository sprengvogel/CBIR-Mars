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
from data import TripletDataset
from CBIRModel import CBIRModel
import hparams as hp
from loss import criterion
import matplotlib.pyplot as plt

# train the model
def train(model, dataloader):
    model.train()
    running_loss = 0.0
    for bi, data in tqdm(enumerate(dataloader), total=int(len(ctx_train) / dataloader.batch_size)):
        anchor = data[0].to(device)
        positive = data[1].to(device)
        negative = data[2].to(device)
        # zero grad the optimizer
        optimizer.zero_grad()
        output_anchor = model(anchor)
        output_pos = model(positive)
        output_neg = model(negative)
        #model.train()
        loss = criterion(output_anchor, output_pos, output_neg)
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
def validate(model, dataloader, epoch):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for bi, data in tqdm(enumerate(dataloader), total=int(len(ctx_val) / dataloader.batch_size)):
            anchor = data[0].to(device)
            positive = data[1].to(device)
            negative = data[2].to(device)

            output_anchor = model(anchor)
            output_pos = model(positive)
            output_neg = model(negative)

            loss = criterion(output_anchor, output_pos, output_neg)
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

    ctx_train = TripletDataset(root="./data/train", transform=data_transform)
    train_loader = torch.utils.data.DataLoader(
        ctx_train,
        batch_size=hp.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    ctx_val = TripletDataset(root="./data/val", transform=data_transform)
    val_loader = torch.utils.data.DataLoader(
        ctx_val, batch_size=hp.BATCH_SIZE, shuffle=True, num_workers=8
    )

    ctx_test = TripletDataset(root="./data/test", transform=data_transform)
    test_loader = torch.utils.data.DataLoader(
        ctx_test, batch_size=hp.BATCH_SIZE, shuffle=False, num_workers=4
    )

    # define device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Computation device: ', device)

    # initialize the model
    model = CBIRModel()
    model.to(device)

    # define optimizer. Also initialize learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=hp.LR, betas=hp.ADAM_BETAS)


    train_loss, val_loss = [], []
    start = time.time()
    best_loss = 1000
    best_epoch = 0

    # start training and validating
    for epoch in range(hp.EPOCHS):
        print(f"Epoch {epoch + 1} of {hp.EPOCHS}")
        train_epoch_loss = train(model, train_loader)
        val_epoch_loss = validate(model, val_loader, epoch)
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
    print(best_epoch)
