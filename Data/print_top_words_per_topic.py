import re

import os

from DataLoading.DataLoader import DataLoader

data_dir = "/home/wlane/PycharmProjects/Kickstarter_classification/Data/kickstarter_corpus.json"
dl = DataLoader(data_dir)
train_data = dl.get_train_data_set()

count_by_label=dict()
for d in train_data:
    label = d['category']
    full_text = d['blurb'] + " "+ d['title'] + " " +d['full_text']
    full_text = full_text.lower()
    zerod_t = re.sub('\d', '0', full_text)
    stripped = re.sub('[.,?\'$%&:;!()\"#@]', "", zerod_t)
    tokens = stripped.split()

    if label not in count_by_label:
        count_by_label[label] = dict()
    for t in tokens:
        if t in count_by_label[label]:
            count_by_label[label][t]+=1
        else:
            count_by_label[label][t]=1

for label, wordcount_d in count_by_label.items():
    sorted_wordcount_d = sorted(wordcount_d.items(), key=lambda x: x[1], reverse=True)
    with open(os.path.join("wordcount_data", label + "-count.txt"), "wb") as f:
        for tup in sorted_wordcount_d:
            f.write(bytes(tup[0] + "\t" + str(tup[1]) + "\n", encoding="utf-8"))


