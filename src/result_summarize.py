import argparse
import os

import numpy as np
import pandas as pd


def get_all_labels(series):
    return list(set(series.values.tolist()))

def calc_accuracy(gt_label, df):
    """
        クエリ一枚あたりに対して、top65(全個体)を列挙した場合に正解かどうかを判定する
        np.arrayを返す
        e.g.) top65のうちk番目で正解 -> 0~k-1まで0, それ以降が1のnp.array
    """
    res = []
    correct = 0
    for db_label in df['db_label']:
        if gt_label == db_label or correct == 1: 
            res.append(1.0)
            correct = 1
        else:
            res.append(0.0)
    return np.array(res)
        
def calc_whole_acc(all_result):
    all_acc = np.zeros(65)
    for label, acc_list in all_result.items():
        all_acc = all_acc + np.array(acc_list)
    all_acc /= 65
    all_acc = all_acc.tolist()
    return all_acc


def main(args):
    df = pd.read_csv(args.result_csv)
    # 1つのクエリにつき65行ずつデータがある
    # 1ラベルあたり65 * 5 = 325行
    
    summary = pd.DataFrame({
        'top_N': list(range(1,66))
    })
    all_result = dict()
    all_labels = get_all_labels(df['gt'])
    for gt_label in all_labels:
        top_N_correct = np.zeros(65) # 個体ごとのtopN正解率を算出
        for i in range(5):
            gt_df = df[(df['gt']==gt_label) & (df['idx_in_query']==i)]
            top_N_correct = top_N_correct + calc_accuracy(gt_label, gt_df)
        top_N_correct /= 5 # クエリは各5枚なので
        top_N_correct = top_N_correct.tolist()
        summary[gt_label] = top_N_correct
        all_result[gt_label] = top_N_correct

    summary['all'] = calc_whole_acc(all_result)
    summary.to_csv(os.path.join("result", args.summary_csv), index=False)
       

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Results Summarization')
    parser.add_argument('--result_csv', type= str, help='the csv file for run summarization')
    parser.add_argument('--summary_csv', type=str, help='the csv file for save summarization results')
    args = parser.parse_args()
    
    main(args)
