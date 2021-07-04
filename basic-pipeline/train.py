import torch
import time
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision import transforms, datasets
from pytorch_metric_learning import losses, miners, distances, reducers
import os
import matplotlib.pyplot as plt

# train the model
def train(model, dataloader):
    model.train()
    running_loss = 0.0
    for bi, data in tqdm(enumerate(dataloader), total=int(len(ctx_train) / dataloader.batch_size)):
        inputs = data[0].to(device)
        labels = data[1].to(device)

        # zero grad the optimizer
        optimizer.zero_grad()
        embeddings = model(inputs)
        embeddings = embeddings.squeeze()
        norm_embeddings = F.normalize(embeddings, p=2, dim=1)

        triplet_indices_tuple = triplet_mining(norm_embeddings, labels)
        triplet_loss = triplet_criterion(norm_embeddings, labels, triplet_indices_tuple)

        #print("triplet loss: ", triplet_loss)
        #print("hash loss: ", hashing_loss)

        loss = triplet_loss

        loss.backward()
        #plot_grad_flow(model.named_parameters())
        # update the parameters
        optimizer.step()
        # add loss of each item (total items in a batch = batch size)
        running_loss += loss.item()

        #if bi == int(hp.EPOCHS*0.95):
        #    triplet_mining.type_of_triplets = "hard"
    final_loss = running_loss / (len(ctx_train) / dataloader.batch_size)

    return final_loss

#validate model
def validate(model, dataloader, epoch):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for bi, data in tqdm(enumerate(dataloader), total=int(len(ctx_val) / dataloader.batch_size)):
            inputs = data[0].to(device)
            labels = data[1].to(device)

            embeddings = model(inputs)
            embeddings = embeddings.squeeze()

            norm_embeddings = F.normalize(embeddings, p=2, dim=1)

            triplet_indices_tuple = triplet_mining(norm_embeddings, labels)
            triplet_loss = triplet_criterion(norm_embeddings, labels, triplet_indices_tuple)

            loss = triplet_loss

            # add loss of each item (total items in a batch = batch size)
            running_loss += loss.item()
    final_loss = running_loss / (len(ctx_val)/ dataloader.batch_size)

    return final_loss


if __name__ == '__main__':

    batch_size = 16
    epochs = 50

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

    ctx_test = datasets.ImageFolder(root="./data/test", transform=data_transform)
    test_loader = torch.utils.data.DataLoader(
        ctx_test, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # define device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Computation device: ', device)

    # initialize the model

    model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]), nn.AvgPool2d(7))
    model.to(device)

    # define loss criterion and optimizer. Also initialize learning rate scheduler
    # define optimizer. Also initialize learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9,0.999))

    distance = distances.LpDistance()#distances.CosineSimilarity()#
    reducer = reducers.ThresholdReducer(low = 0)#DoNothingReducer()#
    triplet_criterion = losses.TripletMarginLoss(margin = 0.2, distance = distance, reducer = reducer)
    triplet_mining = miners.TripletMarginMiner(margin = 0.2, distance = distance, type_of_triplets = "semihard")


    train_loss, val_loss = [], []
    start = time.time()
    best_loss = 1000
    best_epoch = 0

    # start training and validating
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        train_epoch_loss = train(model, train_loader)
        val_epoch_loss = validate(model, val_loader, epoch)
        # save model with best loss
        if val_epoch_loss < best_loss:
            best_epoch = epoch
            best_loss = val_epoch_loss
            print("Saved Model. Best Epoch: " + str(best_epoch))
            torch.save(model.state_dict(), 'outputs/model_best.pth')
        print("Saved last Model.")
        torch.save(model.state_dict(), 'outputs/model_last.pth')
        print(f"Train Loss: {train_epoch_loss}")
        print(f"Val Loss: {val_epoch_loss}")
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
    end = time.time()
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()
    print(f"Finished training in: {((end - start) / 60):.3f} minutes")
    print(f"Best performing Epoch:: {best_epoch}.")
