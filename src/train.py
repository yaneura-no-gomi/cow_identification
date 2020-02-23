import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from termcolor import cprint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import ImageTransform, TripletDataset, make_datapath_list
from models import TripletNet

np.random.seed(seed=5)

def main(args):
    assert args.save_interval % 10 == 0, "save_interval must be a multiple of 10"

    # prepare dirs
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_model, exist_ok=True)
    
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print("Device is", device)

    path_lists = make_datapath_list(root="/home/Share/cow/data/3d_dataset")
    train_path_list, val_path_list = train_val_split(path_lists, val_ratio=args.val_ratio)

    train_dataset = TripletDataset(transform=ImageTransform(), flist=train_path_list)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = TripletDataset(transform=ImageTransform(), flist=val_path_list)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = TripletNet()
    model.to(device)
    
    criterion = nn.MarginRankingLoss(margin=args.margin)

    # choose params to train
    update_params_name = []
    for name, _ in model.named_parameters():
        if 'layer4' in name:
            update_params_name.append(name)
        elif 'fc' in name:
            update_params_name.append(name)

    print("**-----** update params **-----**")
    print(update_params_name)
    print("**-----------------------------**")
    print()

    params_to_update = choose_update_params(update_params_name, model)

    # set optimizer
    optimizer = optim.SGD(params_to_update, lr=1e-4, momentum=0.9)

    # run epoch    
    log_writer = SummaryWriter(log_dir=args.log_dir)
    for epoch in range(args.num_epochs):
        print("-"*80)
        print('Epoch {}/{}'.format(epoch+1, args.num_epochs))

        epoch_loss, epoch_acc = [], []
        for inputs, labels in tqdm(train_dataloader):
            batch_loss, batch_acc = train_one_batch(inputs, labels, model, criterion, optimizer, device)
            epoch_loss.append(batch_loss.item())
            epoch_acc.append(batch_acc.item())
        
        epoch_loss = np.array(epoch_loss)
        epoch_acc = np.array(epoch_acc)
        print('[Loss: {:.4f}], [Acc: {:.4f}] \n'.format(np.mean(epoch_loss), np.mean(epoch_acc)))
        log_writer.add_scalar("train/loss", np.mean(epoch_loss), epoch+1)
        log_writer.add_scalar("train/acc", np.mean(epoch_acc), epoch+1)


        # validation
        if (epoch+1) % 10 == 0:
            print("Run Validation")
            epoch_loss, epoch_acc = [], []
            for inputs, labels in tqdm(val_dataloader):
                batch_loss, batch_acc = validation(inputs, labels, model, criterion, device)
                epoch_loss.append(batch_loss.item())
                epoch_acc.append(batch_acc.item())
            
            epoch_loss = np.array(epoch_loss)
            epoch_acc = np.array(epoch_acc)
            print('[Validation Loss: {:.4f}], [Validation Acc: {:.4f}]'.format(np.mean(epoch_loss), np.mean(epoch_acc)))
            log_writer.add_scalar("val/loss", np.mean(epoch_loss), epoch+1)
            log_writer.add_scalar("val/acc", np.mean(epoch_acc), epoch+1)

            # save model
            if (args.save_interval > 0) and ((epoch+1) % args.save_interval == 0):
                save_path = os.path.join(args.save_model, '{}_epoch_{:.1f}.pth'.format(epoch+1, np.mean(epoch_loss)))
                torch.save(model.state_dict(), save_path)

    log_writer.close()



# ***************************************** #
# *               functions               * #
# ***************************************** #
def train_val_split(path_lists, val_ratio=0.1):
    train_path_list, val_path_list = [], []
    
    for p_list in path_lists:
        for key, value in p_list.items():
            random_plist = random.sample(value, len(value))
            
            val_path_list.append({key:random_plist[:int(len(value)*0.1)]})
            train_path_list.append({key:random_plist[int(len(value)*0.1):]})
    
    return train_path_list, val_path_list

def choose_update_params(update_params_name, model):
    """
        When using pretrained models,
        freeze parameters no need to train
    """
    params_to_update = []

    for name, param in model.named_parameters():
        if name in update_params_name:
            param.requires_grad = True
            params_to_update.append(param)
        else:
            param.requires_grad = False
    
    return params_to_update

def train_one_batch(inputs, labels, model, criterion, optimizer, device):
    """
        Train one batch!
    """
    model.train() # train mode
    inputs_gpu = [i.to(device) for i in inputs]
    dist_anc2pos, dist_anc2neg, embedded_vecs = model(inputs_gpu)
    
    target = torch.FloatTensor(dist_anc2pos.size()).fill_(1).to(device)
    loss = criterion(dist_anc2pos, dist_anc2neg, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    acc = dist_accuracy(dist_anc2pos, dist_anc2neg)

    return loss, acc

def validation(inputs, labels, model, criterion, device):
    
    model.eval() # eval
    with torch.no_grad():
        inputs_gpu = [i.to(device) for i in inputs]
        dist_anc2pos, dist_anc2neg, embedded_vecs = model(inputs_gpu)

    target = torch.FloatTensor(dist_anc2pos.size()).fill_(1).to(device)
    loss = criterion(dist_anc2pos, dist_anc2neg, target)
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
    parser.add_argument('--val_ratio', type=float, default=0.1, help='validation data ratio')
    parser.add_argument('--log_dir', type=str, default='./logs', help='dir path to output logs')
    parser.add_argument('--save_interval', type=int, default=-1, 
        help="epoch interval to save model. Must be a multiple of 10. Do not save if negative value is entered")

    args = parser.parse_args()

    text_c = 'magenta'
    cprint("-" * 50, text_c)
    cprint(("TripletNet Train").center(50), text_c)
    cprint("-" * 50, text_c)

    main(args)
