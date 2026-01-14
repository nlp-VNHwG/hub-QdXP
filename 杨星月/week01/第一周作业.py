import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

from openai import OpenAI

dataset = pd.read_csv('dataset.csv', sep='\t', header=None, nrows=10000)  # 读取数据集
input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))  # 利用jieba对数据集的第一列进行分词并用空格连接起来
vector = CountVectorizer()
vector.fit(input_sentence.values)  # 统计词表，词语不重复
input_feature = vector.transform(input_sentence.values)  # 将文本转换为特征矩阵
model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)  # 训练模型

#  创建客户端
client = OpenAI(
    api_key="sk-33d99039161147798c18f483876dc263",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 第一种方式文本分类（机器学习）
def text_classify_using_ml(text: str) -> str:
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0]

# 第二种方式文本分类（大语言模型）
def text_classify_using_llm(text: str) -> str:
    completion = client.chat.completions.create(
        model="qwen-flash",
        messages=[
            {"role": "user", "content": f"""进行文本分类:{text}
从以下类别中选择，只回答类别：
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
"""},
        ]
    )
    return completion.choices[0].message.content

if __name__ == "__main__":
    print("机器学习: ", text_classify_using_ml("我要去北京天安门"))
    print("大语言模型：", text_classify_using_llm("我要去北京天安门"))
