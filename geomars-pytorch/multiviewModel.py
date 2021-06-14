import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
from data import MultiviewDataset
import hparams as hp
import torch.optim as optim
import math


def multi_criterion(outputs, labels):
    labels = labels.tolist()
    outputs = outputs.tolist()
    L = 0
    for o, out in enumerate(outputs):
        fraction = math.exp(out[labels[o]]) / sum([math.exp(el) for el in out])
        L -= math.log(fraction)

    return torch.tensor((L/len(labels)), dtype=torch.float32, requires_grad=True)


class MultiviewNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(44944, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        return x


if __name__ == '__main__':
    data_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    ctx_train = MultiviewDataset(root="./data/train", transform=data_transform)
    train_loader = torch.utils.data.DataLoader(
        ctx_train,
        batch_size=hp.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    ctx_val = MultiviewDataset(root="./data/val", transform=data_transform)
    val_loader = torch.utils.data.DataLoader(
        ctx_val, batch_size=hp.BATCH_SIZE, shuffle=True, num_workers=8
    )

    ctx_test = MultiviewDataset(root="./data/test", transform=data_transform)
    test_loader = torch.utils.data.DataLoader(
        ctx_test, batch_size=hp.BATCH_SIZE, shuffle=False, num_workers=4
    )

    net = MultiviewNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times
        print(epoch)
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data[0]
            labels = data[2]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = multi_criterion(outputs, labels)
            #loss = criterion(outputs, labels)
            print(loss)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    PATH = 'outputs/view1_net.pth'
    torch.save(net.state_dict(), PATH)