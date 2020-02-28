import glob
import os
import pickle
import random

import pandas as pd
from tqdm import tqdm

random.seed(5)

class Dataset_3D():

    def __init__(self, root="/home/Share/cow/data/3d_dataset"):
        self.root = root
        self.path_list = self.make_datapath_list(root)
        self.train_pl, self.val_pl, self.regi_pl = self.train_val_regi_split(path_lists=self.path_list, 
                                                                             regi_ratio=0.1, val_ratio=0.1)

    def make_datapath_list(self, root="/home/Share/cow/data/3d_dataset"):
        """
        return list : [{label:[paths]},{label:[paths]},...]
        """

        datapath_list = []

        dir_list = os.path.join(root, "*")
        dir_list = sorted(glob.glob(dir_list))
        labels = [os.path.basename(p) for p in dir_list]

        for label in labels:
            datapath_dict = dict()
            target = os.path.join(root, label, "*")
            target_paths = sorted(glob.glob(target))
            datapath_dict[label] = target_paths
            datapath_list.append(datapath_dict)

        return datapath_list

    def train_val_regi_split(self, path_lists, regi_ratio=0.1, val_ratio=0.1):
        train_path_list, regi_path_list, val_path_list = [], [], []
        
        for p_dict in tqdm(path_lists):
            for key, value in p_dict.items():
                random_plist = random.sample(value, len(value))
                # split train(including val) and register
                regi_path_list.append({key:random_plist[:int(len(value)*regi_ratio)]})
                train_val_list = random_plist[int(len(value)*regi_ratio):]
                
                # split train and val
                val_path_list.append({key:train_val_list[:int(len(train_val_list)*val_ratio)]})
                train_path_list.append({key:train_val_list[int(len(train_val_list)*val_ratio):]})
        
        return train_path_list, val_path_list, regi_path_list


class Dataset_RGB():

    def __init__(self, root="/home/Share/cow/data/3d_dataset_test"):
        self.root = root
        self.path_list = self.make_test_data_list(self.root)
        self.train_pl, self.val_pl, self.test_pl = self.train_val_test_split(path_lists=self.path_list, 
                                                                             val_ratio=0.1)

    def make_test_data_list(self, root="/home/Share/cow/data/3d_dataset_test"):
        """
        return list : [{label:[paths]},{label:[paths]},...]
        """

        datapath_list = []

        dir_list = os.path.join(root, "*")
        dir_list = sorted(glob.glob(dir_list))
        labels = [os.path.basename(p) for p in dir_list]

        for label in labels:
            datapath_dict = dict()
            target = os.path.join(root, label, "4k_top_*.jpg")
            target_paths = sorted(glob.glob(target))
            datapath_dict[label] = target_paths
            datapath_list.append(datapath_dict)
        
        return datapath_list

    def train_val_test_split(self, path_lists, val_ratio=0.1):
        train_path_list, test_path_list, val_path_list = [], [], []

        for p_dict in tqdm(path_lists):
            for key, value in p_dict.items():
                random_plist = random.sample(value, len(value))

                # split train(including val) and test(==5)
                test_path_list.append({key:random_plist[:5]})
                train_val_list = random_plist[5:]
                
                # split train and val
                val_path_list.append({key:train_val_list[:int(len(train_val_list)*val_ratio)]})
                train_path_list.append({key:train_val_list[int(len(train_val_list)*val_ratio):]})
        
        return train_path_list, val_path_list, test_path_list


def get_num_of_data(data_list, save_path):
    """
        get the number of test data
        return csv (columns = ids, num_of_data)
    """
    ids = []
    nums = []

    for l in data_list:
        for k, v in l.items():
            ids.append(int(k))
            nums.append(len(v))

    df = pd.DataFrame({
        'id': ids,
        'num_of_data': nums
    }).sort_values('id').reset_index(drop=True)
    # print(df)
    df.to_csv(save_path, index=False)

if __name__ == "__main__":

    data_3d = Dataset_3D()
    data_rgb = Dataset_RGB()

    with open("data/3d_data.pkl", mode='wb') as f:
        pickle.dump(data_3d, f)

    with open("data/rgb_data.pkl", mode='wb') as f:
        pickle.dump(data_rgb, f)