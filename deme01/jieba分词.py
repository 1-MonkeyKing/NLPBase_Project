"""
演示jieba分词库的案例

    分词过程：分词的过程本质上就是找到分词符就行
"""
import jieba

# 精确模式：试图将句子最精确地切开，适合 文本分析
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



# 全模式分词：把句子中所有的词语都扫描出来，速度非常快，但是不能消除歧义,适用于：关键词提取，不需要严格分词的准确性
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
# 搜索引擎模式分词：在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词


# 测试功能代码
if __name__ =='__main__':
    # 测试精确模式
    # demo01()
    # 测试全模式
    demo02()