import torch
import time
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms, datasets
from data import ImageFolderWithLabel, removeclassdoublings
from CBIRModel import CBIRModel
import hparams as hp
from Losses.hashing_loss import hashing_criterion as hashing_criterion
import matplotlib.pyplot as plt
import pickle
from pytorch_metric_learning import losses, miners, distances, reducers
import os
from data import MultiviewDataset
from whitening import WTransform1D, EntropyLoss
from random import sample
from Losses.nt_xent import NT_Xent
from utils import AddGaussianNoise, load_encoder


# train the model

def train(model, dataloader, train_dict, train_dict_view2):
    model.train()
    running_loss = 0.0
    for bi, data in tqdm(enumerate(dataloader), total=int(len(ctx_train) / dataloader.batch_size)):
        cont_loss = 0
        if hp.MULTIVIEWS:
            orig = torch.stack([train_dict[x] for x in data[0]])
            view2 = torch.stack([train_dict_view2[x] for x in data[0]])
            if hp.DOMAIN_ADAPTION:
                orig = source_whitening(orig)
                view2 = source_whitening(view2)

            # zero grad the optimizer
            optimizer.zero_grad()
            embeddings_orig, z_i = model(orig)
            embeddings_view2, z_j = model(view2)
            cont_loss = cont_criterion(z_i, z_j)
        else:
            orig = torch.stack([train_dict[x] for x in data[0]])
            if hp.DOMAIN_ADAPTION:
                orig = source_whitening(orig)
            
            # zero grad the optimizer
            optimizer.zero_grad()
            embeddings_orig = model(orig)
        labels = data[1]


        
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

        if hp.DOMAIN_ADAPTION:
            target_inputs = torch.stack(sample(target_list, len(data[0])))
            target_inputs = target_whitening(target_inputs)
            if hp.MULTIVIEWS:
                target_embeddings, _ = model(target_inputs)
            else:
                target_embeddings = model(target_inputs)
            entropy_loss = entropy_criterion(target_embeddings)
            loss += entropy_loss

        # backpropagation
        loss.backward()
        # update the parameters
        optimizer.step()
        # add loss of each item (total items in a batch = batch size)
        running_loss += loss.item()

    final_loss = running_loss / (len(ctx_train) / dataloader.batch_size)

    return final_loss

#validate model


def validate(model, dataloader,val_dict, val_dict_view2, epoch):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for bi, data in tqdm(enumerate(dataloader), total=int(len(ctx_val) / dataloader.batch_size)):

            cont_loss = 0
            if hp.MULTIVIEWS:
                orig = torch.stack([val_dict[x] for x in data[0]])
                view2 = torch.stack([val_dict_view2[x] for x in data[0]])
                if hp.DOMAIN_ADAPTION:
                    orig = target_whitening(orig)
                    view2 = target_whitening(view2)
                embeddings_orig, z_i = model(orig)
                embeddings_view2, z_j = model(view2)
                cont_loss = cont_criterion(z_i, z_j)
            else:
                orig = torch.stack([val_dict[x] for x in data[0]])
                if hp.DOMAIN_ADAPTION:
                    orig = target_whitening(orig)
                embeddings_orig = model(orig)

            labels = data[1]
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

            if hp.DOMAIN_ADAPTION:
                entropy_loss = entropy_criterion(embeddings_orig)
                loss += entropy_loss

            # add loss of each item (total items in a batch = batch size)
            running_loss += loss.item()
    final_loss = running_loss / (len(ctx_val)/ dataloader.batch_size)

    return final_loss

if __name__ == '__main__':

    torch.cuda.empty_cache()

    data_transform = transforms.Compose(
        [
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
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
        ctx_train_densenet = MultiviewDataset(root="../data/train", transform=multi_transform)
        train_loader_densenet = torch.utils.data.DataLoader(
            ctx_train_densenet,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        ctx_val_densenet = MultiviewDataset(root="../data/val", transform=multi_transform)
        val_loader_densenet = torch.utils.data.DataLoader(
            ctx_val_densenet,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
    else:
        #Use densenet on every image
        ctx_train_densenet = datasets.ImageFolder(root="../data/train", transform=data_transform)
        train_loader_densenet = torch.utils.data.DataLoader(
            ctx_train_densenet,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        ctx_val_densenet = MultiviewDataset(root="../data/val", transform=data_transform)
        val_loader_densenet = torch.utils.data.DataLoader(
            ctx_val_densenet,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
    )

    encoder = load_encoder()
    encoder.requires_grad_(False)
    encoder.eval()
    encoder.to(device)


    train_dict = None
    val_dict = None
    if hp.MULTIVIEWS:
        train_dict_view1 = {}
        train_dict_view2 = {}
        for bi, data in tqdm(enumerate(train_loader_densenet), total=int(len(ctx_train_densenet) / train_loader_densenet.batch_size)):
            view1, view2 = data
            sample_fname, _ = train_loader_densenet.dataset.samples[bi]
            view1 = view1.to(device)
            view2 = view2.to(device)
            output1 = encoder(view1).squeeze().detach().clone()
            output2 = encoder(view2).squeeze().detach().clone()
            train_dict_view1[sample_fname] = output1
            train_dict_view2[sample_fname] = output2

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
    else:
        train_dict = {}

        for bi, data in tqdm(enumerate(train_loader_densenet),
                             total=int(len(ctx_train_densenet) / train_loader_densenet.batch_size)):
            image_data,image_label = data
            sample_fname, _ = train_loader_densenet.dataset.samples[bi]
            image_data = image_data.to(device)
            output = encoder(image_data).squeeze().detach().clone()
            train_dict[sample_fname] = output


        val_dict = {}
        for bi, data in tqdm(enumerate(val_loader_densenet),
                             total=int(len(ctx_val_densenet) / val_loader_densenet.batch_size)):
            image_data, image_label = data
            sample_fname, _ = val_loader_densenet.dataset.samples[bi]
            image_data = image_data.to(device)
            output = encoder(image_data).squeeze().detach().clone()
            val_dict[sample_fname] = output


    ctx_train = ImageFolderWithLabel(root="../data/train", transform=data_transform, interclasstriplets = (hp.INTERCLASSTRIPLETS and not hp.MULTIVIEWS), n_clusters = hp.KMEANS_CLUSTERS, features_dict = train_dict)
    train_loader = torch.utils.data.DataLoader(
        ctx_train,
        batch_size=hp.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=hp.MULTIVIEWS
    )

    ctx_val = ImageFolderWithLabel(root="../data/val", transform=data_transform, interclasstriplets = (hp.INTERCLASSTRIPLETS and not hp.MULTIVIEWS), n_clusters = hp.KMEANS_CLUSTERS, features_dict = val_dict)
    val_loader = torch.utils.data.DataLoader(
        ctx_val,
        batch_size=hp.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=hp.MULTIVIEWS
    )

    ctx_test = ImageFolderWithLabel(root="../data/test", transform=data_transform)
    test_loader = torch.utils.data.DataLoader(
        ctx_test,
        batch_size=hp.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    if hp.DOMAIN_ADAPTION:
        ctx_target_densenet = datasets.ImageFolder(root="../data/database", transform=data_transform)
        target_loader_densenet = torch.utils.data.DataLoader(
            ctx_target_densenet,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        target_list = []
        for bi, data in tqdm(enumerate(target_loader_densenet),
                             total=int(len(ctx_target_densenet) / target_loader_densenet.batch_size)):
            image_data, _ = data
            sample_fname, _ = target_loader_densenet.dataset.samples[bi]
            image_data = image_data.to(device)
            output = encoder(image_data).squeeze().detach().clone()
            target_list.append(output)

        source_whitening = WTransform1D(num_features=hp.DENSENET_NUM_FEATURES, group_size=hp.DA_GROUP_SIZE)
        target_whitening = WTransform1D(num_features=hp.DENSENET_NUM_FEATURES, group_size=hp.DA_GROUP_SIZE)
        source_whitening.to(device)
        target_whitening.to(device)
        entropy_criterion = EntropyLoss()

    # initialize the model
    model = CBIRModel(useEncoder=False, useProjector=hp.MULTIVIEWS)
    model.to(device)

    # define optimizer. Also initialize learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=hp.LR, betas=hp.ADAM_BETAS)

    distance = distances.LpDistance()
    reducer = reducers.ThresholdReducer(low = 0)
    triplet_criterion = losses.TripletMarginLoss(margin = hp.MARGIN, distance = distance, reducer = reducer)

    triplet_mining = miners.TripletMarginMiner(margin=hp.MARGIN, distance=distance, type_of_triplets="semihard")
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
            torch.save(model.state_dict(), 'outputs/model_best.pth')
        print("Saved last Model.")
        torch.save(model.state_dict(), 'outputs/model_last.pth')

        if hp.DOMAIN_ADAPTION:
            torch.save(target_whitening.state_dict(), 'outputs/target_transform.pth')

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
