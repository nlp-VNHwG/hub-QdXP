import pandas as pd
import jieba
from openai import OpenAI
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pkg_resources

datasets = pd.read_csv('dataset.csv', sep='\t', header=None, nrows=10000)
# print(datasets[1].value_counts())

input_sentence = datasets[0].apply(lambda x: " ".join(jieba.lcut(x)))

vectorizer = CountVectorizer()
vectorizer.fit(input_sentence.values)
input_feature = vectorizer.transform(input_sentence.values)

knnmodel = KNeighborsClassifier()
knnmodel.fit(input_feature, datasets[1].values)

svmmodel = SVC()
svmmodel.fit(input_feature, datasets[1].values)


client = OpenAI(
    api_key='sk-f1de23cc9d5e4d2693700c30aeee9765',
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
)

def text_classify_using_ml_knn(text: str) -> str:
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vectorizer.transform([test_sentence])
    return knnmodel.predict(test_feature)[0]

def text_classify_using_ml_svm(text: str) -> str:
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vectorizer.transform([test_sentence])
    return svmmodel.predict(test_feature)[0]

def text_classify_using_llm(text: str) -> str:
    completion = client.chat.completions.create(
        model="qwen-flash",

        messages=[
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
    """},  # 用户的提问
        ]
    )
    return completion.choices[0].message.content

if __name__ ==  '__main__':
    print("大语言模型：", text_classify_using_llm("帮我查询今天的天气"))
    print("机器学习-knn：", text_classify_using_ml_knn("这首歌真好听"))
    print("机器学习-svm: ", text_classify_using_ml_svm("今天是2026年1月22号"))
