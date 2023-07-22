import numpy as np
import pandas as pd

data = pd.read_csv("../data/train.csv")
data = np.array(data)

with open('train.txt', 'w', encoding='utf-8') as fw:
    for line in data:
        idx, title, authors, abstract, key_words, label = line
        if idx == 5000:
            break
        if not isinstance(abstract, str):   # 有数据的 abstract 为空，所以需要特殊处理
            content = title + ' ' + key_words
        else:
            content = title + ' ' + key_words + ' ' + abstract
        fw.write(content + '\t' + str(label) + '\n')

with open('dev.txt', 'w', encoding='utf-8') as fw:
    for line in data:
        idx, title, authors, abstract, key_words, lable = line
        if idx < 5000:
            continue
        if not isinstance(abstract, str):   # 有数据的 abstract 为空，所以需要特殊处理
            content = title + ' ' + key_words
        else:
            content = title + ' ' + key_words + ' ' + abstract
        fw.write(content + '\t' + str(lable) + '\n')

# 测试集
teat_data = pd.read_csv("../data/test.csv")
test_data = np.array(teat_data)

with open('test.txt', 'w', encoding='utf-8') as fw:
    for line in test_data:
        idx, title, authors, abstract, key_words = line
        if not isinstance(abstract, str):   # 有数据的 abstract 为空，所以需要特殊处理
            content = title + ' ' + key_words
        else:
            content = title + ' ' + key_words + ' ' + abstract
        fw.write(content + '\t' + str(0) + '\n')
        
        
        
