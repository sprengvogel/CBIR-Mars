import torch
import time
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from data import MultiviewDataset
import hparams as hp
import matplotlib.pyplot as plt
from SimCLRModel import SimCLR
from loss import criterion
from LARS import LARS


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def contrastive_loss(z_i, z_j):

    # https://github.com/leftthomas/SimCLR/blob/master/main.py

    batch_size = z_i.size()[0]

    temperature = 0.5
    out = torch.cat([z_i, z_j], dim=0)
    # [2*B, 2*B]
    sim_matrix = F.cosine_similarity(out.unsqueeze(1), out.unsqueeze(0), dim=2)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()

    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(z_i * z_j, dim=-1) / temperature)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

    return loss


# train the model
def train(model, dataloader):
    model.train()
    running_loss = 0.0
    for bi, data in tqdm(enumerate(dataloader), total=int(len(ctx_train) / dataloader.batch_size)):
        x_i = data[0].to(device)
        x_j = data[1].to(device)

        z_i, z_j = model(x_i), model(x_j)

        loss = contrastive_loss(z_i, z_j)

        # zero grad the optimizer
        optimizer.zero_grad()
        # backpropagation
        loss.backward()
        # update the parameters
        optimizer.step()
        # add loss of each item (total items in a batch = batch size)
        running_loss += loss.item()
    final_loss = running_loss / len(ctx_train)

    return final_loss


# validate model
def validate(model, dataloader):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for bi, data in tqdm(enumerate(dataloader), total=int(len(ctx_val) / dataloader.batch_size)):
            x_i = data[0].to(device)
            x_j = data[1].to(device)

            z_i, z_j = model(x_i), model(x_j)

            loss = contrastive_loss(z_i, z_j)
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
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k")
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)


if __name__ == '__main__':

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop([224,224]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomApply([AddGaussianNoise(1, 0.5)], 0.5)
    ])

    data_transform = transforms.Compose(
        [
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

    ctx_train = MultiviewDataset(root="./data/trainorig", transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        ctx_train,
        batch_size=hp.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    ctx_val = MultiviewDataset(root="./data/valorig", transform=train_transform)
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

    model = SimCLR(128)
    model.to(device)

    # define optimizer. Also initialize learning rate scheduler
    #optimizer = optim.Adam(model.parameters(), lr=hp.LR, betas=hp.ADAM_BETAS)
    optimizer = LARS(model.parameters(), lr=0.1)

    train_loss, val_loss = [], []
    start = time.time()
    best_loss = 1000
    best_epoch = 0

    # start training and validating
    for epoch in range(hp.EPOCHS):
        print(f"Epoch {epoch + 1} of {hp.EPOCHS}")
        train_epoch_loss = train(model, train_loader)
        val_epoch_loss = validate(model, val_loader)
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
