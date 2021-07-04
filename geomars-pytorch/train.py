from numpy import mod
import torch
import time
import torchvision
from random import randrange
from torch.nn.modules.container import ModuleList
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision import transforms, datasets
from data import ImageFolderWithLabel, removeclassdoublings
from CBIRModel import CBIRModel
import hparams as hp
from loss import criterion as hashing_criterion
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from pytorch_metric_learning import losses, miners, distances, reducers
import os
from data import MultiviewDataset
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


# train the model
def train(model, dataloader, train_dict, train_dict_view2):
    model.train()
    running_loss = 0.0
    for bi, data in tqdm(enumerate(dataloader), total=int(len(ctx_train) / dataloader.batch_size)):
        cont_loss = 0
        if hp.MULTIVIEWS:
            orig = torch.stack([train_dict[x] for x in data[0]])
            view2 = torch.stack([train_dict_view2[x] for x in data[0]])
            optimizer.zero_grad()
            embeddings_orig, z_i = model(orig)
            embeddings_view2, z_j = model(view2)
            cont_loss = cont_criterion(z_i, z_j)
        else:
            orig = torch.stack([train_dict[x] for x in data[0]])
            optimizer.zero_grad()
            embeddings_orig = model(orig)
        labels = data[1]


        # zero grad the optimizer
        # optimizer.zero_grad()
        #embeddings_orig, z_i = model(orig)
        # embeddings_view2, z_j = model(view2)
        norm_embeddings_orig = F.normalize(embeddings_orig, p=2, dim=1)

        if hp.MULTIVIEWS:
            triplet_loss = 0
        else:
            triplet_indices_tuple = triplet_mining(norm_embeddings_orig, labels)
            triplet_loss = triplet_criterion(norm_embeddings_orig, labels, triplet_indices_tuple)

        if hp.INTERCLASSTRIPLETS:
            interclass_labels = data[2]
            inter_class_triplet_indices_tuple = triplet_mining(norm_embeddings_orig, interclass_labels)
            inter_class_triplet_indices_tuple = removeclassdoublings(inter_class_triplet_indices_tuple, labels)
            triplet_loss += triplet_criterion(norm_embeddings_orig, interclass_labels, inter_class_triplet_indices_tuple)

        hashing_loss = hashing_criterion(embeddings_orig)

        loss = triplet_loss + hashing_loss + cont_loss

        # backpropagation
        loss.backward()
        # update the parameters
        optimizer.step()
        # add loss of each item (total items in a batch = batch size)
        running_loss += loss.item()

    final_loss = running_loss / (len(ctx_train) / dataloader.batch_size)

    return final_loss

#validate model
def validate(model, dataloader, val_dict, val_dict_view2, epoch):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for bi, data in tqdm(enumerate(dataloader), total=int(len(ctx_val) / dataloader.batch_size)):
            cont_loss = 0
            if hp.MULTIVIEWS:
                orig = torch.stack([val_dict[x] for x in data[0]])
                view2 = torch.stack([val_dict_view2[x] for x in data[0]])
                #optimizer.zero_grad()
                embeddings_orig, z_i = model(orig)
                embeddings_view2, z_j = model(view2)
                cont_loss = cont_criterion(z_i, z_j)
            else:
                orig = torch.stack([val_dict[x] for x in data[0]])
                #optimizer.zero_grad()
                embeddings_orig = model(orig)
            labels = data[1]
            #embeddings_orig, z_i = model(orig)
            #embeddings_view2, z_j = model(view2)
            #print("labels: ", labels)
            #print(embeddings)
            norm_embeddings_orig = F.normalize(embeddings_orig, p=2, dim=1)
            #norm_embeddings_view2 = F.normalize(embeddings_view2, p=2, dim=1)
            #print(norm_embeddings)
            if hp.MULTIVIEWS:
                triplet_loss = 0
            else:
                triplet_indices_tuple = triplet_mining(norm_embeddings_orig, labels)
                triplet_loss = triplet_criterion(norm_embeddings_orig, labels, triplet_indices_tuple)

            if hp.INTERCLASSTRIPLETS:
                interclass_labels = data[2]
                inter_class_triplet_indices_tuple = triplet_mining(norm_embeddings_orig, interclass_labels)
                inter_class_triplet_indices_tuple = removeclassdoublings(inter_class_triplet_indices_tuple, labels)
                triplet_loss += triplet_criterion(norm_embeddings_orig, interclass_labels, inter_class_triplet_indices_tuple)

            hashing_loss = hashing_criterion(embeddings_orig)

            #cont_loss = contrastive_loss(z_i, z_j)
            #print("Contrastive: " + str(cont_loss))
            #print("Hashing: " + str(hashing_loss))
            #print("Triplet: " + str(triplet_loss))
            #print("triplet loss: ", triplet_loss)
            #print("hash loss: ", hashing_loss)
            loss = triplet_loss + hashing_loss + cont_loss
            #loss = cont_loss
            # add loss of each item (total items in a batch = batch size)
            running_loss += loss.item()
    final_loss = running_loss / (len(ctx_val)/ dataloader.batch_size)

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

    torch.cuda.empty_cache()

    data_transform = transforms.Compose(
        [
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )



    ctx_train = ImageFolderWithLabel(root="./data/train", transform=data_transform, interclasstriplets = hp.INTERCLASSTRIPLETS, n_clusters = hp.KMEANS_CLUSTERS)
    train_loader = torch.utils.data.DataLoader(
        ctx_train,
        batch_size=hp.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    ctx_val = ImageFolderWithLabel(root="./data/val", transform=data_transform, interclasstriplets = hp.INTERCLASSTRIPLETS, n_clusters = hp.KMEANS_CLUSTERS)
    val_loader = torch.utils.data.DataLoader(
        ctx_val,
        batch_size=hp.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    ctx_test = ImageFolderWithLabel(root="./data/test", transform=data_transform)
    test_loader = torch.utils.data.DataLoader(
        ctx_test,
        batch_size=hp.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=True
    )

    # define device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Computation device: ', device)

    if hp.MULTIVIEWS:
        multi_transform = transforms.Compose([
            transforms.RandomResizedCrop([224,224]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomApply([AddGaussianNoise(1, 0.5)], 0.5)
        ])
        #Use densenet on every image
        ctx_train_densenet = MultiviewDataset(root="./data/train", transform=multi_transform)
        train_loader_densenet = torch.utils.data.DataLoader(
            ctx_train_densenet,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        ctx_val_densenet = MultiviewDataset(root="./data/val", transform=multi_transform)
        val_loader_densenet = torch.utils.data.DataLoader(
            ctx_val_densenet,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
    else:
        #Use densenet on every image
        ctx_train_densenet = datasets.ImageFolder(root="./data/train", transform=data_transform)
        train_loader_densenet = torch.utils.data.DataLoader(
            ctx_train_densenet,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        ctx_val_densenet = MultiviewDataset(root="./data/val", transform=data_transform)
        val_loader_densenet = torch.utils.data.DataLoader(
            ctx_val_densenet,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    encoder = torchvision.models.densenet121(pretrained=True)
    #encoder.classifier = nn.Linear(DENSENET_NUM_FEATURES, 15)
    #encoder.load_state_dict(torch.load("densenet121_pytorch_adapted.pth"))
    encoder = torch.nn.Sequential(*(list(encoder.children())[:-1]), nn.AvgPool2d(7))
    encoder.requires_grad_(False)
    encoder.eval()
    encoder.to(device)

    if hp.MULTIVIEWS:
        train_file_path = Path("./train_view1.p")
        if train_file_path.is_file():
            train_dict_view1 = pickle.load(open("train_view1.p","rb"))
            train_dict_view2 = pickle.load(open("train_view2.p","rb"))
        else:
            train_dict_view1 = {}
            train_dict_view2 = {}
            for bi, data in tqdm(enumerate(train_loader_densenet), total=int(len(ctx_train_densenet) / train_loader_densenet.batch_size)):
                #image_data,image_label = data
                #sample_fname, _ = train_loader_densenet.dataset.samples[bi]
                #image_data = image_data.to(device)
                #output = encoder(image_data).squeeze().detach().clone()
                #train_dict[sample_fname] = output
                view1, view2 = data
                sample_fname, _ = train_loader_densenet.dataset.samples[bi]
                view1 = view1.to(device)
                view2 = view2.to(device)
                output1 = encoder(view1).squeeze().detach().clone()
                output2 = encoder(view2).squeeze().detach().clone()
                train_dict_view1[sample_fname] = output1
                train_dict_view2[sample_fname] = output2
            pickle.dump(train_dict_view1, open("train_view1.p", "wb"))
            pickle.dump(train_dict_view2, open("train_view2.p", "wb"))

        val_file_path = Path("./val_view1.p")
        if val_file_path.is_file():
            val_dict_view1 = pickle.load(open("val_view1.p","rb"))
            val_dict_view2 = pickle.load(open("val_view2.p","rb"))
        else:
            val_dict_view1 = {}
            val_dict_view2 = {}
            for bi, data in tqdm(enumerate(val_loader_densenet), total=int(len(ctx_val_densenet) / val_loader_densenet.batch_size)):
                view1, view2 = data
                sample_fname, _ = val_loader_densenet.dataset.samples[bi]
                view1 = view1.to(device)
                view2 = view2.to(device)
                output1 = encoder(view1).squeeze().detach().clone()
                output2 = encoder(view2).squeeze().detach().clone()
                val_dict_view1[sample_fname] = output1
                val_dict_view2[sample_fname] = output2
            pickle.dump(val_dict_view1, open("val_view1.p", "wb"))
            pickle.dump(val_dict_view2, open("val_view2.p", "wb"))
    else:
        train_file_path = Path("./train.p")
        if train_file_path.is_file():
            train_dict = pickle.load(open("train.p", "rb"))
        else:
            train_dict = {}

            for bi, data in tqdm(enumerate(train_loader_densenet),
                                 total=int(len(ctx_train_densenet) / train_loader_densenet.batch_size)):
                image_data,image_label = data
                sample_fname, _ = train_loader_densenet.dataset.samples[bi]
                image_data = image_data.to(device)
                output = encoder(image_data).squeeze().detach().clone()
                train_dict[sample_fname] = output
            pickle.dump(train_dict, open("train.p", "wb"))
        val_file_path = Path("./val.p")
        if val_file_path.is_file():
            val_dict = pickle.load(open("val.p", "rb"))
        else:
            val_dict = {}
            for bi, data in tqdm(enumerate(val_loader_densenet),
                                 total=int(len(ctx_val_densenet) / val_loader_densenet.batch_size)):
                image_data, image_label = data
                sample_fname, _ = val_loader_densenet.dataset.samples[bi]
                image_data = image_data.to(device)
                output = encoder(image_data).squeeze().detach().clone()
                val_dict[sample_fname] = output
            pickle.dump(val_dict, open("val.p", "wb"))
    # initialize the model
    model = CBIRModel(useEncoder=False, useProjector=hp.MULTIVIEWS)
    model.to(device)
    #print(nn.Sequential(*list(model.children())[:-1]))

    # define optimizer. Also initialize learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=hp.LR, betas=hp.ADAM_BETAS)
    #optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    #optimizer = LARS(model.parameters(), lr=0.3 * BATCH_SIZE / 256, weight_decay=0.000006)

    distance = distances.LpDistance()#distances.CosineSimilarity()#
    reducer = reducers.ThresholdReducer(low = 0)#DoNothingReducer()#
    triplet_criterion = losses.TripletMarginLoss(margin = hp.MARGIN, distance = distance, reducer = reducer)
    # contrastive_criterion = losses.ContrastiveLoss()
    triplet_mining = miners.TripletMarginMiner(margin = hp.MARGIN, distance = distance, type_of_triplets = "semihard")
    cont_criterion = NT_Xent(hp.BATCH_SIZE, hp.TEMPERATURE, world_size=1)
    train_loss, val_loss = [], []
    start = time.time()
    best_loss = 1000
    best_epoch = 0

    # start training and validating
    for epoch in range(hp.EPOCHS):
        print(f"Epoch {epoch + 1} of {hp.EPOCHS}")
        if hp.MULTIVIEWS:
            train_epoch_loss = train(model, train_loader, train_dict_view1, train_dict_view2)
            val_epoch_loss = validate(model, val_loader, val_dict_view1, val_dict_view2, epoch)
        else:
            train_epoch_loss = train(model, train_loader, train_dict, False)
            val_epoch_loss = validate(model, val_loader, val_dict, False, epoch)
        # save model with best loss
        if val_epoch_loss < best_loss:
            best_epoch = epoch
            best_loss = val_epoch_loss
            print("Saved Model. Best Epoch: " + str(best_epoch+1))
            #torch.save(nn.Sequential(*list(model.children())[:-1]).state_dict(), 'outputs/model_best.pth')
            torch.save(model.state_dict(), 'outputs/model_best.pth')
        print("Saved last Model.")
        torch.save(model.state_dict(), 'outputs/model_last.pth')
        print(f"Train Loss: {train_epoch_loss}")
        print(f"Val Loss: {val_epoch_loss}")
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
    end = time.time()
    #plt.show()
    print(f"Finished training in: {((end - start) / 60):.3f} minutes")
    print(f"Best performing Epoch:: {best_epoch}.")