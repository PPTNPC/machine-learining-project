import torch
import torch.nn as nn
import jieba
import gensim.models as word2vec
import numpy as np
import csv
import random
from tqdm import tqdm

data_path_train = 'cnews.train.txt'
data_path_test = 'cnews.test.txt'
W2V_path = 'word2vec_new.model'
epochs = 10


class LSTM(nn.Module):
    def __init__(self, input_size=256, hidden_size=16, output_size1=10, num_layers=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 如果当前为抽取强化的词语将会运算
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear1 = nn.Linear(hidden_size, output_size1)

    def forward(self, Input):
        out, _ = self.lstm(Input)
        out = self.linear1(out)
        return out


def read_file(data_path):
    Tag = []
    Including = []
    with open(data_path, encoding="utf-8") as line_reader:
        for line in line_reader:
            tag_sort = line[0] + line[1]
            including_sort = line[3:-1]
            Tag.append(tag_sort)
            Including.append(including_sort)
    return Tag, Including


def including_split(includings):
    spitt_list = []
    for including in range(len(includings)):
        spitt_list.append(list(jieba.cut(includings[including], cut_all=False)))
        if including % 100 == 0:
            print("We have split {} sentences".format(including))
    return spitt_list


def W2V(Tag, Includings, model):
    VIncluding = []
    Vtag = T2Onehot(Tag)
    for word in range(len(Includings)):
        vocab = model.wv
        if Includings[word] in vocab:
            VIncluding.append(model.wv[Includings[word]])
    return torch.tensor(Vtag), torch.tensor(np.array(VIncluding))


def T2Onehot(tag):
    dict_tag = {'体育': 0, '娱乐': 1, '家居': 2, '房产': 3, '教育': 4, '时尚': 5, '时政': 6, '游戏': 7, '科技': 8,
                '财经': 9}
    tag_num = dict_tag[tag]
    tag_list = np.identity(10)[tag_num]
    return tag_list


def write_csv(filename, text):
    with open(filename, "a", encoding="utf-8", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(text)
        print("写入数据成功")
        f.close()


if __name__ == '__main__':
    # 训练文本输入
    tag_train, including_train = read_file(data_path_train)
    len_total = len(including_train)
    including_split_out_train = including_split(including_train)

    # 测试文本输入
    tag_test, including_test = read_file(data_path_test)
    including_split_out_test = including_split(including_test)

    model1 = word2vec.Word2Vec.load(W2V_path)
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
    model = LSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    for epoch in range(epochs):
        things = list(range(len(including_train)))
        random.shuffle(things)
        pbar = tqdm(range(len_total))
        for emus in pbar:
            train_tag, V_including_train = W2V(tag_train[emus], including_split_out_train[emus], model1)
            train_including = V_including_train.to(device, dtype=torch.float)
            train_including = torch.unsqueeze(train_including, 0)
            optimizer.zero_grad()
            outputs = model(train_including)
            outputs = outputs[0][len(outputs) - 1].to(device, dtype=torch.float)
            train_tag = train_tag.to(device, dtype=torch.float)
            loss = criterion(outputs, train_tag)
            loss.backward()
            optimizer.step()
            pbar.set_description("Processing %s" % emus)
            # if best_loss - loss.float() < 1e-6:
            #     break
            # if loss.float() < best_loss:
            #     best_loss = loss.float()
        torch.save(model.state_dict(), 'best_model.pth')
