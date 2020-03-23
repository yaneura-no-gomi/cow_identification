import argparse
import os
import pickle
import random

import numpy as np
import pandas as pd
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
from models import EmbeddingImg, TripletNet
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

    regi_dataloader = DataLoader(regi_dataset, batch_size=1, shuffle=False)

    print("Registering Imgs to DB")
    regi_dict = dict()
    for inputs, labels in tqdm(regi_dataloader):
        # labels_db = labels[0]
        for i, img in enumerate(inputs):
            embedded_imgs = model(img.to(device)) # size -> (1, 1000)
            if i==0:
                embedded_regi = embedded_imgs.detach()
            else:
                embedded_regi = torch.cat((embedded_regi, embedded_imgs.detach()), dim=0)
        regi_dict[labels[0]] = embedded_regi
    return regi_dict


class CalcSimilarity():
    def __init__(self, db, test_dataloader, model):
        self.db = db
        self.test_dataloader = test_dataloader
        self.model = model
        self.query_dict = self.embedding_query(self.model, self.test_dataloader)

    def embedding_query(self, model, test_dataloader):
        query_dict = dict()
        print("embedding test data (query)")
        for inputs, labels in tqdm(test_dataloader):
            for i, img in enumerate(inputs):
                embedded_imgs = model(img.to(device))
                if i==0:
                    embedded_test = embedded_imgs.detach()
                else:
                    embedded_test = torch.cat((embedded_test, embedded_imgs.detach()), dim=0)
            query_dict[labels[0]] = embedded_test
        
        return query_dict

    def euclid_dist(self):
        res_df = pd.DataFrame()
        for gt_label, qs in tqdm(self.query_dict.items()):
            for q_idx, q in enumerate(qs):
                # labelがgt_labelの個体の特徴ベクトルq
                calc_res = []
                for db_label, db_matrix in self.db.items():
                    # db_label: 現在参照しているDB画像群のラベル
                    # db_matrix: 現在参照しているDB画像群の特徴ベクトルをcatしてある行列
                    db_q_diff = db_matrix - q
                    euclid_m = torch.mm(db_q_diff, torch.t(db_q_diff))
                    euclid_dis = torch.diagonal(euclid_m) # <- size [68,] 68枚のdb画像から得た特徴ベクトルとqの距離
                    similarity, min_idx = torch.min(euclid_dis, dim=0) # <- similarity: 類似度、min_idx: その類似度をもったDB画像のインデックス
                    calc_res.append((gt_label, db_label, similarity.item(), q_idx, min_idx.item()))

                calc_res = sorted(calc_res, key=lambda x:x[2]) # similarityをkeyにしてソート
                df = pd.DataFrame(calc_res, columns=['gt', 'db_label', 'similarity', 'idx_in_query', 'idx_in_db'])
                res_df = pd.concat([res_df, df], axis=0).reset_index(drop=True)

        return res_df

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
    else:
        os.makedirs(args.register)
        print("building a new database!")
        db = registering_imgs(regi_dataset, args.batch_size, model)
        with open(os.path.join(args.register, "db.pkl"), "wb") as f:
            pickle.dump(db,f)

    # embedding query imgs
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    sim_calc = CalcSimilarity(db, test_dataloader, model)
    res_df = sim_calc.euclid_dist()
    res_df.to_csv("result/test.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training for TripletNet')
    parser.add_argument('--batch_size', type= int, default=2, help='batch size for input')
    parser.add_argument('--register', type=str, required=True, help='dir path for building a database and save as pkl file')
    parser.add_argument('--weight', type=str, required=True, help='the path for loading pretrained weight')

    args = parser.parse_args()
    main(args)
