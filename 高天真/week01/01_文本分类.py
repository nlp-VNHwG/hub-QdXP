import jieba
import pandas as pd
from openai import OpenAI
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from torchgen.gen_functionalization_type import return_str

"""
先把文件读进来
"""
dataset = pd.read_csv(filepath_or_buffer="dataset.csv",sep="\t",header=None,nrows=10000)


"""
特征提取
"""
input_sentence = dataset[0].apply(lambda x:" ".join(jieba.lcut(x)))
#dataset[0].apply（）把dataset第0列的每一行数据拿出来分别进行处理
#lambda x:" ".join(jieba.lcut(x))。匿名函数，拿出的每行数据作为输入的x，输出为" ".join(jieba.lcut(x)
#jieba.lcut(x) 进行分词返回一个列表 如['我', '爱', '自然语言', '处理']
#" ".join(...)用空格将列表中的词连接起来，返回: "我 爱 自然语言 处理"，因为 CountVectorizer 默认是按空格 / 制表符 / 换行符拆分文本

vector = CountVectorizer() # 词频向量器	相当于 “准备好一个统计词频、生成向量的工具”。

vector.fit(input_sentence.values) #fit()方法返回训练好的CountVectorizer对象本身，但这个对象已经学习到了词汇表。
#fit() 遍历所有文本，统计出所有不重复的词语，生成词汇表。{'我': 0, '爱': 1, '编程': 2, '很': 3, '有趣': 4, '喜欢': 5, '的': 6}

input_feature = vector.transform(input_sentence.values)
#根据 fit 生成的词汇表，把每个文本转换成词频向量（统计每个词语在当前文本中出现的次数），最终生成稀疏矩阵。

"""
用 KNN 分类算法，以 input_feature 作为特征，以 dataset[1] 作为标签，训练一个分类模型 
核心目的是让模型学习 “文本特征” 和 “标签” 之间的对应关系，后续可用来预测新文本的标签。
"""

model = KNeighborsClassifier() #创建一个 K 近邻分类器的实例（相当于 “准备好一个分类模型工具”）
model.fit(input_feature,dataset[1].values) #用训练数据 “喂” 模型，让模型完成训练
#model.fit(特征, 标签)：sklearn 所有模型的通用训练方法，必须传入 “特征矩阵” 和 “标签数组”，且两者行数必须一致（每个特征对应一个标签)
#input_feature：之前生成的文本词频特征矩阵


client = OpenAI(
    api_key= "sk-4266f1ed0cba4267b99eab3d053d7514",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


def text_classify_using_ml(text: str) -> str:
    """
    文本分类(机器学习方法实现)：输入文本完成类别划分
    """
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0] #返回的是数据，取数组的第一个值



def text_classify_using_llm(text:str) -> str:
    """
    文本分类(大语言模型实现)：输入文本完成类别划分
    """

    completion = client.chat.completions.create(
        model = "qwen-flash",  # 模型的代号

        messages= [
                    {"role": "user", "content": f"""帮我进行文本分类：{text}

                    输出的类别只能从如下中进行选择， 除了类别之外下列的类别，请给出最合适的类别。
                    FilmTele-Play            
                    Video-Play               
                    Music-Play              
                    Radio-Listen           
                    Alarm-Update        
                    Travel-Query        
                    HomeAppliance-Control  
                    Weather-Query          
                    Calendar-Query      
                    TVProgram-Play      
                    Audio-Play       
                    Other             
                    """} # 用户的提问
                    ]
    )

    return completion.choices[0].message.content


if __name__ == "__main__": #作用：区分 “脚本直接运行” 和 “脚本被导入为模块” 两种场景
    #直接运行这个脚本，会执行if __name__ == "__main__":
    #把这个脚本作为模块导入，则不执行if __name__ == "__main__":

    print("机器学习：",text_classify_using_ml("麻辣烫哪家最好吃"))
    print("大语言模型: ", text_classify_using_llm("麻辣烫哪家最好吃"))
