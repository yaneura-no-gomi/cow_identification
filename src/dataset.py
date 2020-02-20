import glob
import os
import os.path as osp
import random

import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms as T


class ImageTransform():
    
    def __init__(self, resize):
        self.transforms = T.Compose([
            T.Resize(resize),
            T.ToTensor()
        ])

    def __call__(self, img):
        return self.transforms(img)

def visualize_img(path, resize):
    img = Image.open(path)
    plt.imshow(img)
    plt.savefig('./result/tmp.jpg')
    # plt.show()

    transform = ImageTransform(resize)
    img_transformed = transform(img)
    img_transformed = img_transformed.detach().numpy().transpose((1,2,0))

    plt.imshow(img_transformed)
    plt.savefig("./result/tmp_transformed.jpg")

def make_datapath_list(phase='train'):
    """
    return list : [{label:[paths]},{label:[paths]},...]
    """

    datapath_list = []

    if phase == 'test':
        root = "/home/Share/cow/data/hogehoge" # need to modify

    else:
        root = "/home/Share/cow/data/3d_dataset"

    dir_list = osp.join(root, "*")
    dir_list = sorted(glob.glob(dir_list))
    labels = [osp.basename(p) for p in dir_list]


    for label in labels:
        datapath_dict = dict()
        target = osp.join(root, label, "*")
        target_paths = sorted(glob.glob(target))
        datapath_dict[label] = target_paths
        datapath_list.append(datapath_dict)

    return datapath_list

class TripletDataset(data.Dataset):

    def __init__(self, transform=None, phase='train'):
        self.transform = transform
        self.phase = phase
        self.flist = make_datapath_list(self.phase)

    def __len__(self):
        c = 0
        
        for cow_dict in self.flist:
            c += len(cow_dict.values) # number of paths per indivisual
        
        return c

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
            img = Image.open(img_p)
            img = self.transform(img)
            img_transformed.append(img)

        return img_transformed, labels

        

if __name__ == "__main__":
    # flist = make_datapath_list()
    dataset = TripletDataset(ImageTransform(resize = 100))
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    print(dataset.__getitem__(1))