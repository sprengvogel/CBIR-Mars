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


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def contrastive_loss(z_i, z_j):
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)

    temperature = 0.1
    batch_size = z_i.size()[0]
    negatives_mask = torch.eye(batch_size * 2, batch_size * 2, dtype=bool).float().to(device)

    representations = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    sim_ij = torch.diag(similarity_matrix, batch_size)
    sim_ji = torch.diag(similarity_matrix, -batch_size)
    positives = torch.cat([sim_ij, sim_ji], dim=0)

    nominator = torch.exp(positives / temperature)
    denominator = negatives_mask * torch.exp(similarity_matrix / temperature)

    loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
    loss = torch.sum(loss_partial) / (2 * batch_size)
    return loss


# train the model
def train(model, dataloader, train_dict_view1, train_dict_view2):
    model.train()
    running_loss = 0.0
    for bi, data in tqdm(enumerate(dataloader), total=int(len(ctx_train) / dataloader.batch_size)):
        #inputs = torch.stack([train_dict[x] for x in data[0]])
        labels = data[1]
        orig = torch.stack([train_dict_view1[x] for x in data[0]])
        view2 = torch.stack([train_dict_view2[x] for x in data[0]])
        # inputs = torch.cat([view1, view2])
        # label_range = torch.arange(0, len(data[0]))
        # labels = torch.cat([label_range, label_range])

        #grid_img = torchvision.utils.make_grid(orig[:25], nrow=5)
        #torchvision.utils.save_image(grid_img, "images/batch" + str(epoch) + str(bi) + ".jpg")
        #plt.imsave("images/batch" + str(epoch) + str(bi) + ".jpg", grid_img.permute(1, 2, 0).cpu().numpy())
        # zero grad the optimizer
        optimizer.zero_grad()
        embeddings_orig, z_i = model(orig)
        embeddings_view2, z_j = model(view2)
        #print("labels: ", labels)
        #print(embeddings)
        norm_embeddings_orig = F.normalize(embeddings_orig, p=2, dim=1)
        norm_embeddings_view2 = F.normalize(embeddings_view2, p=2, dim=1)
        #print(norm_embeddings)
        triplet_indices_tuple = triplet_mining(norm_embeddings_orig, labels)
        triplet_loss = triplet_criterion(norm_embeddings_orig, labels, triplet_indices_tuple)

        if hp.INTERCLASSTRIPLETS:
            interclass_labels = data[2]
            inter_class_triplet_indices_tuple = triplet_mining(norm_embeddings_orig, interclass_labels)
            inter_class_triplet_indices_tuple = removeclassdoublings(inter_class_triplet_indices_tuple, labels)
            triplet_loss += triplet_criterion(norm_embeddings_orig, interclass_labels, inter_class_triplet_indices_tuple)

        hashing_loss = hashing_criterion(embeddings_orig)

        cont_loss = contrastive_loss(z_i, z_j)
        #print("triplet loss: ", triplet_loss)
        #print("hash loss: ", hashing_loss)
        loss = triplet_loss + hashing_loss + cont_loss
        #print("loss: ", loss)


        # backpropagation
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
def validate(model, dataloader, val_dict_view1, val_dict_view2, epoch):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for bi, data in tqdm(enumerate(dataloader), total=int(len(ctx_val) / dataloader.batch_size)):
            labels = data[1]
            orig = torch.stack([val_dict_view1[x] for x in data[0]])
            view2 = torch.stack([val_dict_view2[x] for x in data[0]])

            embeddings_orig, z_i = model(orig)
            embeddings_view2, z_j = model(view2)
            #print("labels: ", labels)
            #print(embeddings)
            norm_embeddings_orig = F.normalize(embeddings_orig, p=2, dim=1)
            norm_embeddings_view2 = F.normalize(embeddings_view2, p=2, dim=1)
            #print(norm_embeddings)
            triplet_indices_tuple = triplet_mining(norm_embeddings_orig, labels)
            triplet_loss = triplet_criterion(norm_embeddings_orig, labels, triplet_indices_tuple)

            if hp.INTERCLASSTRIPLETS:
                interclass_labels = data[2]
                inter_class_triplet_indices_tuple = triplet_mining(norm_embeddings_orig, interclass_labels)
                inter_class_triplet_indices_tuple = removeclassdoublings(inter_class_triplet_indices_tuple, labels)
                triplet_loss += triplet_criterion(norm_embeddings_orig, interclass_labels, inter_class_triplet_indices_tuple)

            hashing_loss = hashing_criterion(embeddings_orig)

            cont_loss = contrastive_loss(z_i, z_j)
            #print("Contrastive: " + str(cont_loss))
            #print("Hashing: " + str(hashing_loss))
            #print("Triplet: " + str(triplet_loss))
            #print("triplet loss: ", triplet_loss)
            #print("hash loss: ", hashing_loss)
            loss = triplet_loss + hashing_loss + cont_loss

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
    )

    ctx_val = ImageFolderWithLabel(root="./data/val", transform=data_transform, interclasstriplets = hp.INTERCLASSTRIPLETS, n_clusters = hp.KMEANS_CLUSTERS)
    val_loader = torch.utils.data.DataLoader(
        ctx_val,
        batch_size=hp.BATCH_SIZE,
        shuffle=True,
        num_workers=4
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
    encoder = torchvision.models.densenet121(pretrained=True)
    #encoder.classifier = nn.Linear(DENSENET_NUM_FEATURES, 15)
    #encoder.load_state_dict(torch.load("densenet121_pytorch_adapted.pth"))
    encoder = torch.nn.Sequential(*(list(encoder.children())[:-1]), nn.AvgPool2d(7))
    encoder.requires_grad_(False)
    encoder.eval()
    encoder.to(device)
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

    # initialize the model
    model = CBIRModel(useEncoder=False)
    model.to(device)
    #print(nn.Sequential(*list(model.children())[:-1]))

    # define optimizer. Also initialize learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=hp.LR, betas=hp.ADAM_BETAS)
    #optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    #optimizer = LARS(model.parameters(), lr=0.3 * BATCH_SIZE / 256, weight_decay=0.000006)

    distance = distances.LpDistance()#distances.CosineSimilarity()#
    reducer = reducers.ThresholdReducer(low = 0)#DoNothingReducer()#
    triplet_criterion = losses.TripletMarginLoss(margin = hp.MARGIN, distance = distance, reducer = reducer)
    triplet_mining = miners.TripletMarginMiner(margin = hp.MARGIN, distance = distance, type_of_triplets = "semihard")

    train_loss, val_loss = [], []
    start = time.time()
    best_loss = 1000
    best_epoch = 0

    # start training and validating
    for epoch in range(hp.EPOCHS):
        print(f"Epoch {epoch + 1} of {hp.EPOCHS}")
        train_epoch_loss = train(model, train_loader, train_dict_view1, train_dict_view2)
        val_epoch_loss = validate(model, val_loader, val_dict_view1, val_dict_view2, epoch)
        # save model with best loss
        if val_epoch_loss < best_loss:
            best_epoch = epoch
            best_loss = val_epoch_loss
            print("Saved Model. Best Epoch: " + str(best_epoch+1))
            torch.save(nn.Sequential(*list(model.children())[:-1]).state_dict(), 'outputs/model_best.pth')
        print("Saved last Model.")
        torch.save(nn.Sequential(*list(model.children())[:-1]).state_dict(), 'outputs/model_last.pth')
        print(f"Train Loss: {train_epoch_loss}")
        print(f"Val Loss: {val_epoch_loss}")
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
    end = time.time()
    #plt.show()
    print(f"Finished training in: {((end - start) / 60):.3f} minutes")
    print(f"Best performing Epoch:: {best_epoch}.")