import glob
import os
import random

import pandas as pd

def make_test_data_list(root="/home/Share/cow/data/3d_dataset_test"):
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

def choose_test_data(data_list, save_path):
    """
        choose imgs using as test data
        select 5 randomly per indivisual
    """
    random.seed(5)
    res_dict = {'id':[], 'path':[]}
    for l in data_list:
        chosen_list = []
        for k, v in l.items():
            v_5 = random.sample(v, 5)
            for chosen_path in v_5:
                res_dict['id'].append(int(k))
                res_dict['path'].append(chosen_path)
    df = pd.DataFrame(res_dict).sort_values('id').reset_index(drop=True)
    # print(df)
    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    
    all_test_data = make_test_data_list(root="/home/Share/cow/data/3d_dataset_test")
    choose_test_data(all_test_data, 'data/test_imgs.csv')