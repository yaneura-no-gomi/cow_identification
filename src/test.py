import argparse
import os
import pickle
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

from dataset import TestDataset
from make_data import Dataset_3D, Dataset_RGB
from models import TripletNet, EmbeddingImg
from utils import ImageTransform

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print("Device is", device)

def load_data(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data

def registering_imgs(regi_dataset, batch_size, model):
    """
        Embedding imgs for database
        return the matrix made as stacking embeded vecotors
    """

    regi_dataloader = DataLoader(regi_dataset, batch_size=batch_size, shuffle=False)

    print("Registering Imgs to DB")

    for i, data in enumerate(tqdm(regi_dataloader)):
        inputs = data[0]
        labels = list(data[1])
        embedded_imgs = model(inputs.to(device)) # size -> (batch, 1000)

        if i==0:
            embedded_regi = embedded_imgs
            labels_db = labels
        else:
            embedded_regi = torch.cat((embedded_regi, embedded_imgs), dim=0)
            labels_db += labels

    return embedded_regi, labels_db


def main(args):
    model = EmbeddingImg()
    model.load_state_dict(torch.load(args.weight))
    model.to(device)
    model.eval()

    regi_pl = load_data("data/3d_data.pkl").regi_pl
    test_pl = load_data("data/rgb_data.pkl").test_pl

    regi_dataset = TestDataset(transform=ImageTransform(), flist=regi_pl)
    test_dataset = TestDataset(transform=ImageTransform(), flist=test_pl)

    # loading database or building database
    if os.path.exists(args.register):
        print("loading existing database!")
        with open(os.path.join(args.register,"db.pkl"), "rb") as f:
            db = pickle.load(f)
        with open(os.path.join(args.register,"labels.pkl"), "rb") as f:
            labels_db = pickle.load(f)
    else:
        os.makedirs(args.register)
        print("building a new database!")
        db, labels_db = registering_imgs(regi_dataset, args.batch_size, model)
        with open(os.path.join(args.register, "db.pkl"), "wb") as f:
            pickle.dump(db,f)
        with open(os.path.join(args.register, "labels.pkl"), "wb") as f:
            pickle.dump(labels_db,f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training for TripletNet')
    parser.add_argument('--batch_size', type= int, default=2, help='batch size for input')
    parser.add_argument('--register', type=str, required=True, help='dir path for building a database and save as pkl file')
    parser.add_argument('--weight', type=str, required=True, help='the path for loading pretrained weight')

    args = parser.parse_args()
    main(args)