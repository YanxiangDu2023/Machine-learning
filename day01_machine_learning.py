from scipy.stats import pearsonr
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

# 词频-逆文档频率，Term Frequency-Inverse Document Frequency
import jieba
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# DictVectorizer 是 scikit-learn 中的一个工具类，用于将字典列表转换为矩阵形式，这在机器学习中非常有用，特别是对类别特征进行编码时。
def datasets_demo():

    "sklearn 数据集使用"

    iris = load_iris()
    print("iris dataset:\n",iris)
    print("dataset description\n",iris["DESCR"])
    print("feature_name\n",iris.feature_names)

    #数据集划分
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size =0.2, random_state=22)
    # 在机器学习中，数据集的拆分是随机的。如果不控制这种随机性，每次运行代码时，训练集和测试集的样本都可能不同，结果也会有所差异。
    # random_state
    # 就是用来设置随机数种子，控制数据的随机拆分。
    # 当你设置了
    # random_state（例如
    # random_state = 22），它就固定了数据集拆分的随机性，保证你每次运行代码时，训练集和测试集的内容都相同。这使得结果具有可重复性，方便调试和对比实验结果


    print("训练集的特征值\n",x_train,x_train.shape)



def dict_demo():
    data =[{"city":"bj","temperature":100},{"city":"sh","temperature":60},{"city":"sz","temperature":30}]
    # 实例化一个转换器类
    transfer = DictVectorizer(sparse = False) # 密集矩阵
    transfer = DictVectorizer(sparse = True) # 稀疏矩阵只把非0值表示出来，节省内存
    #具体来说，city特征会被独热编码（One - HotEncoding），将不同的城市（如bj、sh、sz）编码为独立的二进制特征。

    # 调用
    data_new = transfer.fit_transform(data)
    print('data_new\n',data_new)
    print('data_new\n', transfer.feature_names_)



def count_demo():
    data = ["life is short, i like python,","life is too long, i dislike python"]
    # 实例化一个转化器
    transfer = CountVectorizer(stop_words=["is","too"])


    #调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new\n",data_new.toarray())
    data_new.toarray()
    # 将稀疏矩阵转换为密集数组（ndarray），以更直观的方式显示结果。
    print("feature names\n", transfer.get_feature_names_out())

   # 每一行表示一个文本的特征向量，每一列表示一个词汇表中的单词。向量中的值表示该单词在该文本中出现的次数。

    return None


def count_chinese_demo2():


    data = ["支持四种分词模式，精确模式，试图将句子最精确地切开，适合文本分析；; ",
            "全模式，把句子中所有的可以成词的词语都扫描出来, 速度非常快，但是不能解决歧义"]

    # 先分词
    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))

    transfer = CountVectorizer(stop_words=["因为","所以"])

    # 调用fit_transform
    data_final = transfer.fit_transform(data_new)
    print("data_new\n", data_final.toarray())

    # 将稀疏矩阵转换为密集数组（ndarray），以更直观的方式显示结果。
    print("feature names\n", transfer.get_feature_names_out())

    return None

def cut_word(text):

    print(jieba.cut(text))

    generator = " ".join(list(jieba.cut(text)))

    # jieba.cut()是中文分词的重要工具，它返回一个生成器，可以逐个产出分词后的词。
    # 使用list()将生成器转换为列表，可以更方便地查看分词结果。
    # ''.join()可以将分词后的列表组合为字符串

    # print(generator)
    return generator

def tfidf_demo():
    data = ["支持四种分词模式，精确模式，试图将句子最精确地切开，适合文本分析；; ",
            "全模式，把句子中所有的可以成词的词语都扫描出来, 速度非常快，但是不能解决歧义"]

    # 先分词
    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))

    transfer = TfidfVectorizer(stop_words=["因为", "所以"])

    # 调用fit_transform
    data_final = transfer.fit_transform(data_new)
    print("data_new\n", data_final.toarray())

    # 将稀疏矩阵转换为密集数组（ndarray），以更直观的方式显示结果。
    print("feature names\n", transfer.get_feature_names_out())

    return None


def minmax_demo():

    data = pd.read_csv("/Users/amandadu/Desktop/coding/machine learning/xiday1/02-代码/dating.txt")
    data = data.iloc[:,:3]
    # data = data.iloc[:, :3]：选择数据集中的前3列。iloc是pandas的方法，用于基于位置选择数据。[:,:3] 表示选择所有行，前三列
    print("data:\n", data)

    # 获取数据
    # 实例化一个转换器类

    transfer = MinMaxScaler(feature_range=(2,3))

    # 调用 fit
    data_new = transfer.fit_transform(data)
    print("data_new\n",data_new)
    return None



def stand_demo():
    data = pd.read_csv("/Users/amandadu/Desktop/coding/machine learning/xiday1/02-代码/dating.txt")
    data = data.iloc[:, :3]
    # data = data.iloc[:, :3]：选择数据集中的前3列。iloc是pandas的方法，用于基于位置选择数据。[:,:3] 表示选择所有行，前三列
    print("data:\n", data)

    transfer = StandardScaler()

    data_new = transfer.fit_transform(data)
    print("data_new\n", data_new)
    return None


def variance_demo():
    # 过滤低方差
    # 低方差特征过滤是一种特征选择方法，用于从数据集中去除方差很小的特征。它的主要思路是：当某个特征在整个数据集中变化很小（即方差接近于 0），意味着它对样本的区分能力很弱，因此可以去除这些特征，简化模型，减少计算复杂度。
    # 方差的含义
    data = pd.read_csv("/Users/amandadu/Desktop/coding/machine learning/xiday1/02-代码/factor_returns.csv")

    data = data.iloc[:,1:-2]

    # 实例化一个转化器
    transfer = VarianceThreshold(threshold=5)

    # threshold = 0.0：表示默认删除方差为0的特征。即那些在数据集中没有变化的特征（所有样本中的取值都相同）。
    # 在特征选择和数据处理的上下文中，特征可以理解为 Excel 文件中的每一列。


    data_new = transfer.fit_transform(data)

    print("variance_demo\n", data_new.shape)

    # 计算某两个变量之间的相关系数
    r1 = pearsonr(data["pe_ratio"], data["pb_ratio"])
    print("相关系数：\n", r1)
    r2 = pearsonr(data['revenue'], data['total_expense'])
    print("revenue与total_expense之间的相关性：\n", r2)

    return None



def pca_demo():

    data = [[2,8,4,5],[6,3,0,8],[5,4,9,1]]

    transfer = PCA(n_components =2)
    # 四个特征降为两个特征

    data_new = transfer.fit_transform(data)

    print("data_new\n",data_new)

    return None







if __name__ == "__main__":
    datasets_demo()
    dict_demo()
    count_demo()
    count_chinese_demo2()
    cut_word("我爱北京天安门")
    tfidf_demo()
    minmax_demo()
    variance_demo()
    pca_demo()
