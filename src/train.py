import argparse

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
    print("device is ", device)

    train_path_list = make_datapath_list(root="/home/Share/cow/data/3d_dataset")
    train_dataset = TripletDataset(transform=ImageTransform(), flist=train_path_list)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = TripletNet()
    model.to(device)
    model.train()

    print(model)

    criterion = nn.MarginRankingLoss(margin=args.margin)
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training for TripletNet')
    parser.add_argument('--batch_size', type= int, default=8, help='batch size for input')
    parser.add_argument('--num_epochs', type= int, default=200, help='the number of epochs')
    parser.add_argument('--margin', type=float, default=0.2, help='margin for triplet loss (default: 0.2)')
    parser.add_argument('--save_model', type=str, default="./weights", help='the directory path for saving trained weight')

    args = parser.parse_args()
    main(args)
