from builtins import bytes, range
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn.manifold import TSNE
import gensim
import gensim.models as word2vec
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname="SimHei.ttf",size=10)
 
 
def tsne_plot(model, words_num):
 
    labels = []
    tokens = []
    #取词向量
    for word in model.wv.index_to_key:
        tokens.append(model.wv[word])
        labels.append(word)
    #通过TSNE方法进行降维perplexity为有效邻居数 
    #n_components 指定了嵌入空间的维度数
    #init 参数决定了初始化低维嵌入的方式pca为主成分分析方式
    #n_iter指定了优化过程中迭代最大次数
    #random_state为随机数生成器种子
    tsne_model = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000, random_state=23)
    #保存数据
    new_values = tsne_model.fit_transform(np.array(tokens))
    x = []
    y = []
    #绘制散点图
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    plt.figure(figsize=(10, 10))
    plt.rcParams['font.sans-serif']=['SimHei']           #''' 添加中文字体'''
    plt.rcParams['axes.unicode_minus']=False  
    for i in range(words_num):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],fontproperties=font,xy=(x[i], y[i]),xytext=(5, 2),textcoords='offset points',ha='right',va='bottom')
    plt.savefig('visual2.png')
    plt.show()
 
if __name__ == '__main__':
    #读取模型
    model = word2vec.Word2Vec.load('word2vec_new.model')
    
    #选取展示的词向量数量
    print(f'There are {len(model.wv.index_to_key)} words in vocab')
    word_num = int(input('please input how many words you want to plot:'))
    #进行可视化
    tsne_plot(model, word_num)