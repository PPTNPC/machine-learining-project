# -*- coding: utf-8 -*-
import logging
import sys
import gensim.models as word2vec
import torch
from gensim.models.word2vec import LineSentence, logger
import jieba


def train_word2vec(dataset_path, out_vector):
    # 设置输出日志
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    # 把语料变成句子集合
    sentences = LineSentence(dataset_path)
    # 训练word2vec模型（size为向量维度，window为词向量上下文最大距离，min_count需要计算词向量的最小词频）
    model = word2vec.Word2Vec(sentences, vector_size=256, sg=1, window=4, min_count=2, workers=5, epochs=10)
    # (iter随机梯度下降法中迭代的最大次数，sg为1是Skip-Gram模型)
    # 保存word2vec模型（创建临时文件以便以后增量训练）
    model.save("word2vec_new.model")
    model.wv.save_word2vec_format(out_vector, binary=False)


# 加载模型
def load_word2vec_model(w2v_path):
    model = word2vec.Word2Vec.load(w2v_path)
    return model


# 计算词语的相似词
def calculate_most_similar(model, word):
    similar_words = model.wv.most_similar(word, topn=10)
    print(word)
    for term in similar_words:
        print(term[0], term[1])


if __name__ == '__main__':
    # dataset_path = "sohu_data_vector.txt" #取数据集
    # out_vector = 'corpusSegDone.vector'   #设定输出
    # train_word2vec(dataset_path, out_vector)#通过word2vec方式训练模型

    model = load_word2vec_model("word2vec_new.model")
    print(model.wv.index_to_key)
    vocab = model.wv
    # vocab_index = model.wv.index_to_key
    # y_temp = [0.0] * 30
    # y_temp[vocab_index.index('中国')] = 1.0
    # print(vocab_index.index('中国'))
    # print(y_temp)
    # x = []
    # x.append(' '.join(jieba.cut(
    #     "韩国执政党国民力量在2024年第22届国会议员选举中遭遇了惨败。最大在野党共同民主党及其卫星政党共赢得了175个席位，继续保持国会第一大党地位，而国民力量党及其卫星政党仅获得108席。",
    #     cut_all=False)))
    # print(x)
    # for words in x:
    #     if words in vocab:
    #         print(words)
    test_list = []
    in_list = input("请输入一段文本：")
    test_list.append(list(jieba.cut(in_list, cut_all=False)))
    print(test_list)

    VIncluding_temp = []
    for word in range(len(test_list[0])):
        vocab = model.wv
        print(test_list[0][word] + '\n')
        if test_list[0][word] in vocab:
            VIncluding_temp.append(model.wv[test_list[0][word]])
            print(test_list[0][word], vocab[test_list[0][word]])
    # inputs = input("shuru")
    # inputs = inputs + "is"
    # print(inputs)
    # similar_words = model.wv.most_similar(zero, topn=1)
    # for term in similar_words:
    #     print(term[0], term[1])
    # x = torch.zeros(1, 2, 120)
    # print(len(x[0,1, :]))
