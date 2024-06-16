本文件夹中包括项目设计的全部源代码，生成模型和增量训练所用的数据集，以及生成和训练后的模型文件以下是各文件的简介

## 源代码

word2vector.py：生成模型源代码，实现功能包括根据数据集生成模型和对模型简单测试

CNN.py：增量训练的源代码，实现功能为通过训练集对模型进行训练

visualization.py：可视化源代码，实现功能为对生成的模型中词向量进行降维并生成散点图

## 数据集

sohu_data_vector.txt：搜狗数据集，用于生成模型

cnews.train.txt：由THUCNews数据集划分的训练数据集，用于增量训练

## 模型

word2vec_new.model：由gensim中Word2Vec部分生成的模型

best_model.pth：由pytroch神经网络增量训练的模型



visual.png：由word2vec_new.model中三百个词向量生成的散点图