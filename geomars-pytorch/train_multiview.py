import torch
import time
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from data import MultiviewDataset
from CBIRModel import CBIRModel
import hparams as hp
from loss import criterion
import matplotlib.pyplot as plt
import numpy as np
from multiviewModel import MultiviewNet
import math


def computeV(Qmc):

    V = np.sum(np.abs(Qmc - np.mean(Qmc)))

    return V / len(Qmc)


def computeE(view1, view2):
    E = []
    num_images = len(view1)
    Q = np.zeros((hp.NUM_VIEWS, num_images, hp.NUM_CLASSES))
    Q[0, :, :] = view1
    Q[1, :, :] = view2

    for n in range(num_images):
        En = []
        for m in range(hp.NUM_VIEWS):
            result = 0
            # c_index = np.argmax(Q[m, :, :], axis=1)
            # c_max = np.amax(Q[m, :, :], axis=1)
            # c_max_index = np.argmax(c_max)
            V = []
            V_sum = 0
            for c in range(hp.NUM_CLASSES):
                # V.append(computeV(Q[m, :, c]))
                # V_sum = math.sqrt(computeV(Q[m, :, c]))
                V.append(computeV(Q[m, :, c]))
                V_sum = math.sqrt(computeV(Q[m, :, c]))
            result += math.sqrt(max(V))
            result -= V_sum / num_images
            En.append(result)
        E.append(En)

    E = np.asarray(E)
    for n in range(num_images):
        for m in range(hp.NUM_VIEWS):
            E[n][m] = math.log(E[n][m] + np.abs(np.min(E[n])) + 1) / math.log(np.max(np.abs(E[n])) + np.abs(np.min(E[n])) + 1)
    return E


def computeLm(b1, b2, y):
    a = 0.5
    b1 = b1.detach().cpu().numpy()
    b2 = b2.detach().cpu().numpy()
    Lm = -(y-1) * np.max(a - np.linalg.norm(b1-b2), 0) \
         + (y+1) * np.linalg.norm(b1-b2) \
         + a * (np.linalg.norm(np.abs(b1)-1, ord=1) + np.linalg.norm(np.abs(b2)-1, ord=1))
    return Lm

def computeL(E, B1, B2, labels):
    L = 0
    B = [B1, B2]
    for n in range(len(B1)):
        for m in range(hp.NUM_VIEWS):
            L += E[n][m]
            for i in range(len(B1)):
                y = -1
                if labels[n] == labels[i]:
                    y = 1
                L += computeLm(B[m][n], B[m][i], y)
    return 0

# train the model
def train(model, dataloader, densenet):
    model.train()
    running_loss = 0.0
    for bi, data in tqdm(enumerate(dataloader), total=int(len(ctx_train) / dataloader.batch_size)):
        view1 = data[0].to(device)
        view2 = data[1].to(device)
        labels = data[2].to(device)
        # zero grad the optimizer
        optimizer.zero_grad()

        dense_view1 = densenet(view1)
        dense_view2 = densenet(view2)

        E = computeE(dense_view1, dense_view2)

        output_view1 = model(dense_view1)
        output_view2 = model(dense_view2)

        L = computeL(E, output_view1, output_view2, labels)

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
def validate(model, dataloader, epoch, densenet):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for bi, data in tqdm(enumerate(dataloader), total=int(len(ctx_val) / dataloader.batch_size)):
            view1 = data[0].to(device)
            view2 = data[1].to(device)
            negative = data[2].to(device)

            dense_view1 = densenet(view1)
            dense_view2 = densenet(view2)

            output_anchor = model(dense_anchor)
            output_pos = model(dense_positive)
            output_neg = model(dense_negative)

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

    # define device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Computation device: ', device)

    # initialize the model

    # densenet = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)
    # #densenet = nn.Sequential(*list(densenet.children())[:-1])
    # densenet.fc = nn.Linear(1000, len(ctx_train.classes))
    # densenet.to(device)
    # densenet.requires_grad_(False)
    # densenet.eval()
    densenet = MultiviewNet()
    path = 'outputs/view1_net.pth'
    densenet.load_state_dict(torch.load(path))
    densenet.requires_grad_(False)


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
        train_epoch_loss = train(model, train_loader, densenet)
        val_epoch_loss = validate(model, val_loader, epoch, densenet)
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
