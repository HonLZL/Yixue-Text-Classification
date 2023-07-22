import copy
import time
from importlib import import_module

import pandas as pd
import torch.nn.functional as F
import numpy as np
import torch
from sklearn import metrics

from utils import get_time_dif, build_iterator, build_dataset


def inference(config, model, test_iter):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    predict_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, _ in test_iter:
            outputs = model(texts)
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            predict_all = np.append(predict_all, predic)
    return predict_all

def find_bug(infer):
    with open("yixue/no_name_data/test.txt", 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        labels = []
        for line in lines:
            line = line.split('\t')
            label = line[1].split()
            labels.append(int(label[0]))
        infer = np.array(infer)
        labels = np.array(labels)
        index = np.arange(0, len(infer))
        diff = index[labels != infer]
        print(labels[:20])
        print(infer[:20])
        print("*********************")
        print(diff)
        for i in diff:
            print(lines[i])


if __name__ == '__main__':
    x = import_module('models.bert')
    config = x.Config('yixue')
    train_data, dev_data, test_data = build_dataset(config)
    test_iter = build_iterator(test_data, config)

    model = x.Model(config).to(config.device)
    res = inference(config, model, test_iter)

    # find_bug(res)

    test_data = pd.read_csv("yixue/data/test.csv")
    test_data = np.array(test_data)
    key_words = []
    for line in test_data:
        key_words.append(line[4])
    uuid = range(2358)
    sub = pd.DataFrame()
    sub = sub.assign(uuid=uuid, Keywords=key_words, label=res)
    sub.to_csv("sub_best_acc.csv", index=False)
    print(sub)

