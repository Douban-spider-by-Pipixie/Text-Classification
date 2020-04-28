import jieba
from numpy import *
import pickle  # 持久化
import os
from sklearn.feature_extraction.text import TfidfTransformer  # TF-IDF向量转换类
from sklearn.feature_extraction.text import TfidfVectorizer  # TF_IDF向量生成类
from sklearn.datasets.base import Bunch
from sklearn.naive_bayes import MultinomialNB  # 多项式贝叶斯算法
from sklearn.externals import joblib #模型保存与读取
def readFile(path):
    with open(path, 'r', errors='ignore') as file:  # 文档中编码有些问题，所有用errors过滤错误
        content = file.read()
        file.close()
        return content

def saveFile(path, result):
    with open(path, 'w', errors='ignore') as file:
        file.write(result)
        file.close()

def segText(inputPath, resultPath):
    fatherLists = os.listdir(inputPath)  # 主目录
    for eachDir in fatherLists:  # 遍历主目录中各个文件夹
        eachPath = inputPath + eachDir + "/"  # 保存主目录中每个文件夹目录，便于遍历二级文件
        each_resultPath = resultPath + eachDir + "/"  # 分词结果文件存入的目录
        if not os.path.exists(each_resultPath):
            os.makedirs(each_resultPath)
        childLists = os.listdir(eachPath)  # 获取每个文件夹中的各个文件
        for eachFile in childLists:  # 遍历每个文件夹中的子文件
            eachPathFile = eachPath + eachFile  # 获得每个文件路径
            #  print(eachFile)
            content = readFile(eachPathFile)  # 调用上面函数读取内容
            # content = str(content)
            result = (str(content)).replace("\r\n", "").strip()  # 删除多余空行与空格
            # result = content.replace("\r\n","").strip()

            cutResult = jieba.cut(result)  # 默认方式分词，分词结果用空格隔开
            saveFile(each_resultPath + eachFile, " ".join(cutResult))  # 调用上面函数保存文件

def bunchSave(inputFile, outputFile):
    catelist = os.listdir(inputFile)
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    bunch.target_name.extend(catelist)  # 将类别保存到Bunch对象中
    for eachDir in catelist:
        eachPath = inputFile + eachDir + "/"
        fileList = os.listdir(eachPath)
        for eachFile in fileList:  # 二级目录中的每个子文件
            fullName = eachPath + eachFile  # 二级目录子文件全路径
            bunch.label.append(eachDir)  # 当前分类标签
            bunch.filenames.append(fullName)  # 保存当前文件的路径
            bunch.contents.append(readFile(fullName).strip())  # 保存文件词向量
    with open(outputFile, 'wb') as file_obj:  # 持久化必须用二进制访问模式打开
        pickle.dump(bunch, file_obj)
        # pickle.dump(obj, file, [,protocol])函数的功能：将obj对象序列化存入已经打开的file中。
        # obj：想要序列化的obj对象。
        # file:文件名称。
        # protocol：序列化使用的协议。如果该项省略，则默认为0。如果为负值或HIGHEST_PROTOCOL，则使用最高的协议版本

def readBunch(path):
    with open(path, 'rb') as file:
        bunch = pickle.load(file)
        # pickle.load(file)
        # 函数的功能：将file中的对象序列化读出。
    return bunch


def writeBunch(path, bunchFile):
    with open(path, 'wb') as file:
        pickle.dump(bunchFile, file)


def getStopWord(inputFile):
    stopWordList = readFile(inputFile).splitlines()
    return stopWordList

def getTFIDFMat(inputPath, stopWordList, outputPath):  # 求得TF-IDF向量
    bunch = readBunch(inputPath)
    tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[],
                       vocabulary={})
    '''读取tfidfspace'''
    # 初始化向量空间
    vectorizer = TfidfVectorizer(stop_words=stopWordList, sublinear_tf=True, max_df=0.5)
    transformer = TfidfTransformer()  # 该类会统计每个词语的TF-IDF权值
    # 文本转化为词频矩阵，单独保存字典文件
    tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
    tfidfspace.vocabulary = vectorizer.vocabulary_  # 获取词汇
    '''over'''
    writeBunch(outputPath, tfidfspace)

def getTestSpace(testSetPath, trainSpacePath, stopWordList, testSpacePath):
    bunch = readBunch(testSetPath)
    # 构建测试集TF-IDF向量空间
    testSpace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[],
                      vocabulary={})
    '''
       读取testSpace
       '''
    # 导入训练集的词袋
    trainbunch = readBunch(trainSpacePath)
    # 使用TfidfVectorizer初始化向量空间模型  使用训练集词袋向量
    vectorizer = TfidfVectorizer(stop_words=stopWordList, sublinear_tf=True, max_df=0.5,
                                 vocabulary=trainbunch.vocabulary)
    transformer = TfidfTransformer()
    testSpace.tdm = vectorizer.fit_transform(bunch.contents)
    testSpace.vocabulary = trainbunch.vocabulary
    # 持久化
    writeBunch(testSpacePath, testSpace)

if __name__ == '__main__':
    datapath = "./data/"  #原始数据路径
    stopWord_path = "./stop/stopword.txt"#停用词路径
    test_path = "./test/"            #测试集路径
    '''
    以上三个文件路径是已存在的文件路径，下面的文件是运行代码之后生成的文件路径
    dat文件是为了读取方便做的，txt文件是为了给大家展示做的，所以想查看分词，词频矩阵
    词向量的详细信息请查看txt文件，dat文件是通过正常方式打不开的
    '''
    test_split_dat_path =  "./test_set.dat" #测试集分词bat文件路径
    testspace_dat_path ="./testspace.dat"   #测试集输出空间矩阵dat文件
    train_dat_path = "./train_set.dat"  # 读取分词数据之后的词向量并保存为二进制文件
    tfidfspace_dat_path = "./tfidfspace.dat"  #tf-idf词频空间向量的dat文件
    test_split_path = './split/test_split/'  # 测试集分词路径
    split_datapath = "./split/split_data/"  # 对原始数据分词之后的数据路径

 # #输入训练集    
    #  #    segText(datapath,#读入数据
    #  #            split_datapath)#输出分词结果
    #  #    bunchSave(split_datapath,#读入分词结果
    #  #              train_dat_path)  # 输出分词向量
    #  #    stopWordList = getStopWord(stopWord_path)  # 获取停用词表
    #  #    getTFIDFMat(train_dat_path, #读入分词的词向量
    #  #                stopWordList,    #获取停用词表
    #  #                tfidfspace_dat_path) #tf-idf词频空间向量的dat文件
    '''
    测试集的每个函数的参数信息请对照上面的各个信息，是基本相同的
    '''
    #输入测试集
    stopWordList = getStopWord(stopWord_path)  # 获取停用词表
    segText(test_path,test_split_path)  # 对测试集读入文件，输出分词结果
    bunchSave(test_split_path,test_split_dat_path)  #
    getTestSpace(test_split_dat_path,
                 tfidfspace_dat_path,
                 stopWordList,
                 testspace_dat_path)
    #ayesAlgorithm(tfidfspace_dat_path,testspace_dat_path)

    trainSet = readBunch(tfidfspace_dat_path)
    testSet = readBunch(testspace_dat_path)
    clf = MultinomialNB(alpha=0.001).fit(trainSet.tdm, trainSet.label)
    # alpha:0.001 alpha 越小，迭代次数越多，精度越高

    #joblib.dump(clf,"train_model.m")  #模型保存
    clf = joblib.load("train_model.m")  #模型加载

    #训练过一次模型后就可以直接调用模型进行预测了

    '''处理结束'''
    predicted = clf.predict(testSet.tdm)
    total = len(predicted)
    rate = 0
    for flabel, fileName, expct_cate in zip(testSet.label, testSet.filenames, predicted):
        print(fileName, ":实际类别：", flabel, "-->预测类别：", expct_cate)
        if flabel != expct_cate:
            rate += 1
    print("error rate:", float(rate) * 100 / float(total), "%")