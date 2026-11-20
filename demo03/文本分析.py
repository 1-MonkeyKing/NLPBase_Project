"""
案例:
    演示 文本分析的常见操作.

文本分析的作用:
    帮助我们理解数据语料, 快速检查出语料中可能存在的问题.
    例如:
        数据质量类:        错别字, 语法错误, 重复内容, 缺失值, 噪声数据...
        分布不均衡问题:    标签分布不均, 句子长度不同...

文本分析的方式:
    1. 标签的数量分布
    2. 句子长度分布
    3. 词频统计和关键字词云
"""

# 导包
import jieba                            # 分词
import seaborn as sns                   # 画图
import pandas as pd                     # 数据处理包
import matplotlib.pyplot as plt         # 画图包，推荐使用
from itertools import chain             # 迭代器工具
import jieba.posseg as pseg             # 词性标注包
from wordcloud import WordCloud        # 词云包

# 解决中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']          # 如果是Mac本, 可以换成: 'Arial Unicode MS'
plt.rcParams['axes.unicode_minus'] = False

# 实现：训练集和测试集 标签分布 的可视化统计
def dm01_label_sns_countplot():
    # 1、设置 538 风格（不做也行）
    plt.style.use("fivethirtyeight")

    # 2、读取训练集和 测试集
    """
    参数1：文件路径            参数2：列分隔符（csv文件用, 而tsv文件用的\t）
    """
    train_data = pd.read_csv("data/train.tsv",sep="\t")
    dev_data = pd.read_csv("data/dev.tsv",sep="\t")
    # print(train_data.head())

    # 3、统计训练集标签的 0（负样本）和1（正样本）的数量，并且可视化（采用计数柱状图）
    """
    参数1：x轴标签    参数2：数据集     参数3：用于分组的分类变量      # 参数4：是否显示图例（默认为True）
    """
    sns.countplot(x="label",data=train_data,hue="label",legend=False)
    plt.title("train_label")    # 设置标题
    plt.tight_layout()          # 紧凑布局
    plt.show()

    # 4、统计测试集标签的 0（负样本）和1（正样本）的数量，并且可视化（采用计数柱状图）
    sns.countplot(x="label", data=dev_data, hue="label", legend=False)
    plt.title("dev_label")  # 设置标题
    plt.tight_layout()  # 紧凑布局
    plt.show()


# 实现：训练集和测试集的 句子长度分布的 可视化统计
def dm02_len_sns_distplot():
    # 1、读取训练集和测试集
    train_data = pd.read_csv("data/train.tsv",sep="\t")
    dev_data = pd.read_csv("data/dev.tsv", sep="\t")

    # 2、计算训练集每个句子的长度
    # 思路1： train_data["sentence_length"] = list(map(函数对象的功能（获取句子的长度），句子))
    train_data["sentence_length"] = list(map(lambda x:len(x),train_data["sentence"]))
    dev_data["sentence_length"] = list(map(lambda x:len(x),dev_data["sentence"]))
    # 思路2：train_data["sentence_length"] = train_data["sentence"].apply(lambda x:len(x))
    #print(train_data.head())

    # 3、绘制训练集的句子长度分布
    # 图一：计数柱状图
    sns.countplot(x="sentence_length",data=train_data)
    plt.title("训练集的句子长度分布计数柱状图")
    plt.xticks([])      # 隐藏x轴的刻度显示
    plt.show()
    # 图二：密度曲线图
    # 参数3 是加上一个曲线
    sns.histplot(x="sentence_length",data=train_data,kde=True)
    plt.title("训练集的句子长度分布密度曲线图")
    plt.show()

    # 4、绘制测试集的句子长度分布
    # 图一：计数柱状图
    sns.countplot(x="sentence_length", data=dev_data)
    plt.title("测试集的句子长度分布计数柱状图")
    plt.xticks([])  # 隐藏x轴的刻度显示
    plt.show()
    # 图二：密度曲线图
    # 参数3 是加上一个曲线
    sns.histplot(x="sentence_length", data=dev_data, kde=True)
    plt.title("测试集的句子长度分布密度曲线图")
    plt.show()


# 训练集和验证集样本长度散点图
def dm03_sns_stripplot():
    # 1、设置 538 风格（不做也行）
    plt.style.use("fivethirtyeight")

    # 2、读取训练集和 测试集
    train_data = pd.read_csv("data/train.tsv", sep="\t")
    dev_data = pd.read_csv("data/dev.tsv", sep="\t")

    # 3 获取 训练集 数据长度列：train_data['sentence_length'] = list(map(lambda x:len(x) , ...))
    train_data['sentence_length'] = list(map(lambda x: len(x), train_data['sentence']))

    # 4 获取 测试集 数据长度列：dev_data['sentence_length'] = list(map(lambda x:len(x) , ...))
    dev_data['sentence_length'] = list(map(lambda x: len(x), dev_data['sentence']))

    # 5、统计正负样本散点分布
    # 训练集
    # 参1：x轴标签 参2：y轴标签 参3：数据集  参4：用于分组
    sns.stripplot(x="label",y="sentence_length",data=train_data,hue="label")
    plt.title("训练集正负样本长度散点图")
    plt.show()
    # 测试集
    # 参1：x轴标签 参2：y轴标签 参3：数据集    参4：用于分组
    sns.stripplot(x="label",y="sentence_length",data=dev_data,hue="label")
    plt.title("测试集正负样本长度散点图")
    plt.show()

if __name__ == "__main__":
    # 测试训练集和测试集分布的可视化
    # dm01_label_sns_countplot()

    # 测试训练集和测试集 句子长度的分布
    # dm02_len_sns_distplot()

    # 训练集和验证集样本长度散点图
    dm03_sns_stripplot()