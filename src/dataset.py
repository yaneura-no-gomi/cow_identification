import random

import torch.utils.data as data
from PIL import Image


class TripletDataset(data.Dataset):

    def __init__(self, transform=None, flist=None):
        self.transform = transform
        self.flist = flist

    def __len__(self):
        # c = 0
        # for cow_dict in self.flist:
        #     c += len(list(*cow_dict.values())) # number of paths per indivisual
        # return c
        return len(self.flist)

    def __getitem__(self, idx):
        """
        At first, pick up the anchor, and positive randomly from same path list.
        Negative image is randomly chosen from the other path list.
        """

        anchor_dict = self.flist[idx]
        for label, path_list in anchor_dict.items():
            anchor_label = label
            anchor_paths = path_list

        anchor = random.choice(anchor_paths)

        pos_label = anchor_label
        pos = anchor
        while pos != anchor:
            pos = random.choice(anchor_paths)

        neg_label = anchor_label
        while neg_label == anchor_label:
            neg_dict = random.choice(self.flist)

            for key, path_list in neg_dict.items():
                neg_label = key
                neg_paths = path_list
        neg = random.choice(neg_paths)

        labels = [anchor_label, pos_label, neg_label]

        img_transformed = []
        for img_p in [anchor, pos, neg]:
            # print(img_p)
            img = Image.open(img_p)
            img = self.transform(img)
            img_transformed.append(img)

        return img_transformed, labels
