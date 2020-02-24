import glob
import os.path as osp

from PIL import Image
from torchvision import transforms as T
import matplotlib.pyplot as plt


class ImageTransform():
    def __init__(self, resize=224):
        self.transforms = T.Compose([
            T.Resize(resize),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # for using pre-trained ResNet
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
    img_transformed = img_transformed.detach().numpy().transpose((1, 2, 0))

    plt.imshow(img_transformed)
    plt.savefig("./result/tmp_transformed.jpg")


def make_datapath_list(root="/home/Share/cow/data/3d_dataset"):
    """
    return list : [{label:[paths]},{label:[paths]},...]
    """

    datapath_list = []

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
