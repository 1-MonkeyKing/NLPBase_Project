"""
演示jieba分词库的案例

    分词过程：分词的过程本质上就是找到分词符就行
"""
import jieba

# ①精确模式：试图将句子最精确地切开，适合 文本分析
def demo01():
    # 1、定义待分词的文本
    content = "传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能"

    # 2、使用jieba进行精确分词（默认模式）->cut_all=False
    result1 = jieba.cut(content,cut_all=False)  # result1是一个生成器对象，只能遍历一次

    # 3、从生成器对象中获取所有的分词结果
     # 3.1 next()函数逐个获取
    # print(next(result1))
     # 3.2 遍历方式
    # for item in result1:
    #     print(item)
     # 3.3 使用列表的方式输出
      # 3.3.1 把生成器转成列表
    # list1 = list(result1)
    # print(f"list1:{list1}")
      # 3.3.2 切词时直接返回list就行
    list2 = jieba.lcut(content,cut_all=False)
    print(f"list2:{list2}")



# ②全模式分词：把句子中所有的词语都扫描出来，速度非常快，但是不能消除歧义,适用于：关键词提取，不需要严格分词的准确性
def demo02():
    # 1、定义待分词的文本
    content = "传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能"

    # 2、使用jieba进行全模式分词->cut_all=True
    result1 = jieba.cut(content,cut_all=True)  # result1是一个生成器对象，只能遍历一次

    # 3、从生成器对象中获取所有的分词结果
     # 3.1 next()函数逐个获取
    # print(next(result1))
     # 3.2 遍历方式
    # for item in result1:
    #     print(item)
     # 3.3 使用列表的方式输出
      # 3.3.1 把生成器转成列表
    # list1 = list(result1)
    # print(f"list1:{list1}")
      # 3.3.2 切词时直接返回list就行
    list2 = jieba.lcut(content,cut_all=True)
    print(f"list2:{list2}")



# ③搜索引擎模式分词：在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词、文本匹配
"""
解释：
    搜索引擎分词模式 → 在精确模式分词的基础上，对长词进行再次切分，提高召回率。

例如：

    场景1：用户录入 “程序员”
        精确模式：　　只能匹配包含完整 “程序员” 的文档。
        搜索引擎模式：不仅能匹配 “程序员” 文档，还能匹配 “程序”、“员” 的文档，提高召回。

    场景2：实际应用场景（电商搜索），商品标题为：<<苹果手机保护套>>，用户搜索 <<苹果套>>
        精确模式：　　无法匹配，分词为：("苹果", "手机", "保护套")
        搜索引擎模式：能匹配，分词为：("苹果", "手机", "保护", "套")
"""
def demo03():
    # 1、定义待分词的文本
    content = "传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能"

    # 2、使用jieba进行搜索引擎模式分词->cut_for_search
    result1 = jieba.cut_for_search(content)  # result1是一个生成器对象，只能遍历一次

    # 3、从生成器对象中获取所有的分词结果
    # 3.1 next()函数逐个获取
    # print(next(result1))
    # 3.2 遍历方式
    # for item in result1:
    #     print(item)
    # 3.3 使用列表的方式输出
    # 3.3.1 把生成器转成列表
    # list1 = list(result1)
    # print(f"list1:{list1}")
    # 3.3.2 切词时直接返回list就行
    list2 = jieba.lcut_for_search(content)
    print(f"list2（搜索引擎模式）:{list2}")

    # 4、打印 精确模式分词、全模式分词，以便上述的 搜索引擎模式分词进行对比
    list3 = jieba.lcut(content,cut_all = False)
    print(f"list3（精确模式）:{list3}")
    list4 = jieba.lcut(content, cut_all=True)
    print(f"list4（全模式）:{list4}")



# ④jieba支持中文繁体分词（针对中国香港、澳门、台湾地区）
def demo04():
    # 中文繁体分词
    content = "煩惱即是菩提，我暫且不提"
    mydata = jieba.lcut(content,cut_all=False)
    print('mydata-->', mydata)


# ⑤演示jieba分词中的自定义词典，适用于稍微生僻一点的或者有特殊要求的词，一般不会太多
"""
添加自定义词典后，jieba 能够准确识别出词典中出现的词汇，提升整体的识别准确率。
词典（文件的）格式，每一行分 3 个部分，分别是：
    词语（必选） 词频（可省略） 词性（可省略）        中间用空格隔开，顺序不能颠倒
"""
def demo05():
    # 1、定义待分词的文本
    content = "传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能"

    #2、执行未使用用户自定义词典的分词
    list1 = jieba.lcut(content,cut_all = False)
    print(f"list1（未使用用户自定义词典）:{list1}")

    #3、加载用户自定义的词典
    jieba.load_userdict("../data/userdict.txt")

    # 4、执行用户自定义词典的分词
    list2 = jieba.lcut(content,cut_all = False)
    print(f"list2（户自定义词典）:{list2}")



# 测试功能代码
if __name__ =='__main__':
    # 测试精确模式
    # demo01()

    # 测试全模式
    #demo02()

    # 测试搜索引擎模式
    #demo03()

    # 测试jieba繁体字分词
    #demo04()

    # 测试用户自定义词典
    demo05()
