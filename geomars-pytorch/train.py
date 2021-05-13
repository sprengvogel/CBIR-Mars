import torch
import time
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision.utils import save_image
from torchvision import transforms, datasets


# train the model
def train(model, dataloader):
    model.train()
    running_loss = 0.0
    for bi, data in tqdm(enumerate(dataloader), total=int(len(ctx_train) / dataloader.batch_size)):
        image_data = data[0].to(device)
        label = data[1].to(device)
        # zero grad the optimizer
        optimizer.zero_grad()
        outputs = model(image_data)
        model.train()
        loss = criterion(outputs, label)
        # backpropagation
        loss.backward()
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
            image_data = (data[0].to(device))
            label = data[1].to(device)

            outputs = model(image_data)
            loss = criterion(outputs, label)
            # add loss of each item (total items in a batch = batch size)
            running_loss += loss.item()
            # calculate batch psnr (once every `batch_size` iterations)
        outputs = outputs.cpu()
        save_image(outputs, f"Outputs/val_sr{epoch}.png")
    final_loss = running_loss / len(ctx_val)

    return final_loss


if __name__ == '__main__':

    batch_size = 16
    num_classes = 15
    epochs = 2

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
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    model.to(device)

    # define loss criterion and optimizer. Also initialize learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)


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
            torch.save(model.state_dict(), 'Outputs/model_best.pth')
        print(f"Train Loss: {train_epoch_loss}")
        print(f"Val Loss: {val_epoch_loss}")
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
    end = time.time()
    print(f"Finished training in: {((end - start) / 60):.3f} minutes")
    print(best_epoch)