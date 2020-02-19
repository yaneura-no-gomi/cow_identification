import os
import os.path as osp
import glob

import torch
from torchvision import transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import torch.utils.data as data

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

def make_datapath_dict(phase='train'):
    """
    return dict : dict[label] -> paths of the cow which has the label
    """

    if phase == 'test':
        root = "/home/Share/cow/data/hogehoge" # need to modify

    else:
        root = "/home/Share/cow/data/3d_dataset"

    dir_list = osp.join(root, "*")
    dir_list = sorted(glob.glob(dir_list))
    labels = [osp.basename(p) for p in dir_list]

    datapath_dict = dict()

    for label in labels:
        target = osp.join(root, label, "*")
        target_paths = sorted(glob.glob(target))
        datapath_dict[label] = target_paths

    return datapath_dict

class TripletDataset(data.Dataset):

    def __init__(self, transform=None, phase='train'):
        self.transform = transform
        self.phase = phase
        self.fdict = make_datapath_dict(self.phase)

    def __len__(self):
        c = 0
        
        for _, i in self.dict.items():
            c += len(i)
        
        return c

    def __getitem__(self, idx):
        """
        At first, pick up the anchor, and positive randomly from same directory.
        Negative image is randomly chosen from the other directory images.
        """

        

if __name__ == "__main__":
    