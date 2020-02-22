import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ImageTransform, TripletDataset, make_datapath_list
from models import TripletNet


def main(args):
    
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print("Device is", device)

    train_path_list = make_datapath_list(root="/home/Share/cow/data/3d_dataset")
    train_dataset = TripletDataset(transform=ImageTransform(), flist=train_path_list)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = TripletNet()
    model.to(device)
    model.train() # train mode

    criterion = nn.MarginRankingLoss(margin=args.margin)

    # choose params to train
    update_params_name = []
    for name, _ in model.named_parameters():
        if 'layer4' in name:
            update_params_name.append(name)
        elif 'fc' in name:
            update_params_name.append(name)

    print("**-----update params-----**")
    print(update_params_name)
    print("**-----------------------**")
    print()

    params_to_update = choose_update_params(update_params_name, model)

    # set optimizer
    optimizer = optim.SGD(params_to_update, lr=1e-4, momentum=0.9)

    for epoch in range(args.num_epochs):
        print("--------------------------------------------------------------")
        print('Epoch {}/{}'.format(epoch+1, args.num_epochs))

        epoch_loss, epoch_acc = [], []
        print(len(train_dataloader))
        for inputs, labels in tqdm(train_dataloader):
            batch_loss, batch_acc = train_one_batch(inputs, labels, model, criterion, optimizer, args.margin, device)
            epoch_loss.append(batch_loss.item())
            epoch_acc.append(batch_acc.item())
        
        epoch_loss = np.array(epoch_loss)
        epoch_acc = np.array(epoch_acc)
        print('[Loss: {:.4f}], [Acc: {:.4f}]'.format(np.mean(epoch_loss), np.mean(epoch_acc)))


def choose_update_params(update_params_name, model):
    """
        When using pretrained models,
        freeze parameters no need to training
    """
    params_to_update = []

    for name, param in model.named_parameters():
        if name in update_params_name:
            param.requires_grad = True
            params_to_update.append(param)
        else:
            param.requires_grad = False
    
    return params_to_update

def train_one_batch(inputs, labels, model, criterion, optimizer, margin, device):
    """
        Train one epoch!
    """
    
    inputs_gpu = [i.to(device) for i in inputs]
    dist_anc2pos, dist_anc2neg, embedded_vecs = model(inputs_gpu)
    
    target = torch.FloatTensor(dist_anc2pos.size()).fill_(1).to(device)
    loss = criterion(dist_anc2pos, dist_anc2neg, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    acc = dist_accuracy(dist_anc2pos, dist_anc2neg)

    return loss, acc


def dist_accuracy(dist_anc2pos, dist_anc2neg):
    """
     if dist_anc2neg > dist_anc2pos, that pred is correct.
     Counting correct preds and calculate accuracy per batch
    """
    margin = 0
    pred = (dist_anc2neg - dist_anc2pos - margin).cpu().data
    return (pred > 0).sum()*1.0/dist_anc2neg.size()[0]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training for TripletNet')
    parser.add_argument('--batch_size', type= int, default=8, help='batch size for input')
    parser.add_argument('--num_epochs', type= int, default=200, help='the number of epochs')
    parser.add_argument('--margin', type=float, default=0.5, help='margin for triplet loss (default: 0.5)')
    parser.add_argument('--save_model', type=str, default="./weights", help='the directory path for saving trained weight')

    args = parser.parse_args()
    main(args)
