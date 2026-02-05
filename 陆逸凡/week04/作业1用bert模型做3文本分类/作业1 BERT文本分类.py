import pandas as pd  # 数据处理库
import torch  # PyTorch深度学习框架
from sklearn.model_selection import train_test_split  # 数据集划分工具
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
# BertTokenizer: BERT分词器，将文本转换为模型可理解的token
# BertForSequenceClassification: 用于文本分类的BERT模型
# Trainer: Hugging Face提供的训练器，简化训练过程
# TrainingArguments: 训练参数配置类

from sklearn.preprocessing import LabelEncoder  # 标签编码器，将文本标签转换为数字
from datasets import Dataset  # Hugging Face数据集类
import numpy as np  # 数值计算库

import logging
import sys
import os
import time

'''
作业：
重新找一个公开数据集， 或 直接标注一个文本分类数据集（推荐3个以上的类别个数），
复现加载bert base 模型在新数据集上的微调过程。最终需要输入一个新的样本进行测试，
验证分类效果是否准确。
'''

# 基本配置 - 输出到文件和控制台
log_file_path = os.path.join(os.path.dirname(__file__), 'app.log')
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 日志格式
    handlers=[
        logging.FileHandler(log_file_path, mode='a', encoding='utf-8'),  # 输出到文件
        logging.StreamHandler(sys.stdout)  # 输出到控制台
    ]
)

# 使用日志
logger = logging.getLogger(__name__)

# 加载和预处理数据
# 当前文件所在目录的上级目录
train_file_path = os.path.join(os.path.dirname(__file__), 'ag_news_3class/train.csv')
#把文件中的反斜杠替换为正斜杠，确保路径格式一致
train_file_path = os.path.normpath(train_file_path).replace('\\', '/')
logger.info(f"训练数据文件路径: {train_file_path}")
# 从CSV文件读取数据，指定列名和表头
dataset_df = pd.read_csv(train_file_path, 
                        names=['text', 'label', 'label_name'],  # 指定列名：文本、数字标签、文本标签
                        header=0)  # 第一行作为表头

# 初始化 LabelEncoder，用于将文本标签转换为数字标签
lbl = LabelEncoder()
# 拟合数据并转换前1500个标签，得到数字标签
# fit_transform: 先学习标签到数字的映射，然后应用转换
labels = lbl.fit_transform(dataset_df['label_name'].values[:1500])
# 提取前1500个文本内容
texts = list(dataset_df['text'].values[:1500])

# 分割数据为训练集和测试集
# train_test_split: 将数据随机分割为训练集和测试集
x_train, x_test, train_labels, test_labels = train_test_split(
    texts,             # 文本数据
    labels,            # 对应的数字标签
    test_size=0.2,     # 测试集比例为20%，训练集80%
    stratify=labels    # 确保训练集和测试集的标签分布一致，保持类别比例
)

# 从预训练模型加载分词器和模型
# from_pretrained: 加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('D:/AI/bert-base-chinese')  # 加载中文BERT分词器
# 加载中文BERT模型，设置为三分类任务
model = BertForSequenceClassification.from_pretrained('D:/AI/bert-base-chinese', num_labels=3)

# 使用分词器对训练集和测试集的文本进行编码
# tokenizer: 将文本转换为模型输入格式
# truncation=True: 如果文本过长则截断，保留前64个token
# padding=True: 对齐所有序列长度，填充到最长序列的长度
# max_length=64: 最大序列长度为64个token
train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=64)

# 将编码后的数据和标签转换为 Hugging Face `datasets` 库的 Dataset 对象
# Dataset: Hugging Face的数据集格式，便于与Trainer配合使用
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],           # 文本的token ID序列，每个ID对应一个词或子词
    'attention_mask': train_encodings['attention_mask'], # 注意力掩码，1表示实际token，0表示填充token
    'labels': train_labels                               # 对应的数字标签
})
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],            # 测试集token ID
    'attention_mask': test_encodings['attention_mask'],  # 测试集注意力掩码
    'labels': test_labels                                # 测试集标签
})

# 定义用于计算评估指标的函数
def compute_metrics(eval_pred):
    """
    计算模型评估指标
    
    参数:
    eval_pred: Trainer传递的元组，包含(logits, labels)
    logits: 模型输出的原始预测值，shape为[batch_size, num_labels]
    labels: 真实标签，shape为[batch_size]
    
    返回:
    字典，包含评估指标名称和值
    """
    # eval_pred 是一个元组，包含模型预测的 logits 和真实的标签
    logits, labels = eval_pred
    # 找到 logits 中最大值的索引，即预测的类别
    # argmax: 返回最大值所在的索引，axis=-1表示在最后一个维度（类别维度）上操作
    predictions = np.argmax(logits, axis=-1)
    # 计算预测准确率并返回一个字典
    # (predictions == labels).mean(): 计算预测正确的比例
    return {'accuracy': (predictions == labels).mean()}

# 配置训练参数
# TrainingArguments: 训练相关的超参数设置
training_args = TrainingArguments(
    output_dir='./results',              # 训练输出目录，用于保存模型、日志和检查点
    num_train_epochs=10,                 # 训练的总轮数，每个epoch会完整遍历一次训练集
    per_device_train_batch_size=8,       # 训练时每个设备（GPU/CPU）的批次大小
    per_device_eval_batch_size=8,        # 评估时每个设备的批次大小
    warmup_steps=100,                    # 学习率预热的步数，学习率从0逐渐增加到设定值，有助于稳定训练初期
    learning_rate=2e-5,                  # 初始学习率，BERT微调的常用学习率
    weight_decay=0.01,                   # 权重衰减，L2正则化系数，用于防止过拟合
    logging_dir='./logs',                # 日志存储目录
    logging_steps=50,                    # 每隔50步记录一次日志
    eval_strategy="epoch",               # 评估策略，"epoch"表示每个epoch结束后进行评估
    eval_steps=50,                       # 每50步评估一次（当eval_strategy="steps"时有效）
    save_strategy="epoch",               # 模型保存策略，"epoch"表示每个epoch结束后保存模型
    load_best_model_at_end=True,         # 训练结束后加载效果最好的模型
    metric_for_best_model="accuracy",    # 指定最佳模型的评判标准为准确率
    greater_is_better=True,              # 准确率越高越好
    save_total_limit=2,                  # 最多保存2个模型检查点，自动删除旧的
    report_to="none",                    # 禁用wandb、tensorboard等报告工具
)

# 实例化 Trainer 简化模型训练代码
# Trainer: Hugging Face提供的训练器，封装了训练循环、评估、保存等功能
trainer = Trainer(
    model=model,                         # 要训练的模型
    args=training_args,                  # 训练参数配置
    train_dataset=train_dataset,         # 训练数据集
    eval_dataset=test_dataset,           # 评估数据集
    compute_metrics=compute_metrics,     # 用于计算评估指标的函数
)

# 深度学习训练过程：数据获取，epoch batch 循环，梯度计算 + 参数更新

# 开始训练模型
logger.info("开始训练模型...")
logger.info(f"训练参数: {training_args}")
start_time = time.time()
# train(): 启动训练过程，包含前向传播、损失计算、反向传播、参数更新
train_result = trainer.train()
# 记录训练结束时间
end_time = time.time()
training_duration = end_time - start_time
# 提取训练指标
train_metrics = train_result.metrics
logger.info(f"训练完成，耗时: {training_duration:.2f}秒")
logger.info("训练指标:")
for key, value in train_metrics.items():
    if isinstance(value, float):
        logger.info(f"  {key}: {value:.4f}")
    else:
        logger.info(f"  {key}: {value}")

# ================= 2. 评估过程日志记录 =================
logger.info("开始评估模型...")
# 在测试集上进行最终评估
# evaluate(): 在评估数据集上评估模型性能
eval_result = trainer.evaluate()

logger.info("评估结果:")
for key, value in eval_result.items():
    if key == 'eval_loss':
        logger.info(f"评估损失: {value:.4f}")
    elif key == 'eval_accuracy':
        logger.info(f"评估准确率: {value:.4f} ({value*100:.2f}%)")
    elif key == 'eval_runtime':
        logger.info(f"评估耗时: {value:.2f}秒")
    elif key == 'eval_samples_per_second':
        logger.info(f"评估速度: {value:.2f} 样本/秒")
    else:
        logger.info(f"{key}: {value}")

# Trainer是比较简单，适合训练过程比较规范化的模型
# 如果需要定制化训练过程（如特殊的损失函数、优化器、训练逻辑），Trainer可能无法满足

def predict_news_category(text, model, tokenizer, label_encoder,testing_label_name):
    """
    使用训练好的模型预测新闻类别
    
    参数:
    text: 要预测的新闻文本
    model: 训练好的BERT模型
    tokenizer: BERT分词器
    label_encoder: 标签编码器，用于数字标签和文本标签的转换
    
    返回:
    字典，包含预测结果和相关信息
    """
    # 对输入文本进行编码
    # return_tensors='pt': 返回PyTorch张量
    encoding = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=64)
    
    # 将模型设置为评估模式
    # eval(): 关闭dropout、batch normalization的训练模式，使用训练好的参数
    model.eval()
    correct = 0
    with torch.no_grad():  # 不计算梯度，节省内存和计算资源
        # 获取模型的输出
        # **encoding: 解包字典，相当于model(input_ids=..., attention_mask=...)
        outputs = model(**encoding)
        logits = outputs[0]  # 模型输出的原始分数，第一个元素通常是logits
        # 计算预测的类别ID
        # argmax(): 找到logits中最大值的位置，即最可能的类别
        predicted_class_id = logits.argmax().item()
        # 计算预测的概率分布
        # softmax(): 将logits转换为概率，所有概率和为1
        # dim=1: 在第1个维度（类别维度）上应用softmax
        # squeeze(): 移除维度为1的维度
        probabilities = torch.softmax(logits, dim=1).squeeze()
        # 再次获取预测类别（与上面相同，用于一致性检查）
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        # 将数字标签转换回原始文本标签
        # inverse_transform(): 将数字标签转换为原始文本标签
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]
        if predicted_label == testing_label_name:
            correct = 1
    
 
    
    # 返回预测结果字典
    return {
        'text': text,  # 原始文本
        'predicted_label': predicted_label,  # 预测的文本标签
        'predicted_class': predicted_class,  # 预测的数字标签
        'probabilities': probabilities,  # 每个类别的概率
        'confidence': probabilities[predicted_class].item(),  # 预测类别的置信度（概率值）
        'correct': correct
        
    }



# 测试文本列表，包含不同主题的新闻
logger.info("开始批量预测测试...")
# 加载和预处理数据
# 当前文件所在目录的上级目录
test_file_path = os.path.join(os.path.dirname(__file__), 'ag_news_3class/test.csv')
#把文件中的反斜杠替换为正斜杠，确保路径格式一致
test_file_path = os.path.normpath(test_file_path).replace('\\', '/')
logger.info(f"测试数据文件路径: {test_file_path}")
# 从CSV文件读取数据，指定列名和表头
dataset_df = pd.read_csv(test_file_path, 
                        names=['text', 'label', 'label_name'],  # 指定列名：文本、数字标签、文本标签
                        header=0)  # 第一行作为表头


# 初始化 LabelEncoder，用于将文本标签转换为数字标签
lb2 = LabelEncoder()

start =0
size = 100
total_acuuracy =[]
epoch_size = 10
#查看10轮的正确率
for i in range (epoch_size):
    # 拟合数据并转换前100个标签，得到数字标签
    # fit_transform: 先学习标签到数字的映射，然后应用转换
    test_labels = lb2.fit_transform(dataset_df['label_name'].values[start:start+size])
    # 提取前100个文本内容
    test_texts = list(dataset_df['text'].values[start:start+size])
    testing_label_name = list(dataset_df['label_name'].values[start:start+size])
    correct = 0
    total = len(test_texts)
    # 对每个测试文本进行预测并打印结果
    for index, text in enumerate(test_texts):
        # 调用预测函数
        result = predict_news_category(text, model, tokenizer, lb2,testing_label_name[index])
        if result['correct'] == 1:
            correct += 1
        # 打印预测结果
        # [:50]: 只显示前50个字符，避免输出过长
        logger.info(f"文本: {text[:50]}...")
        logger.info(f"预测类别: {result['predicted_label']},实际类别: {testing_label_name[index]}")
        # :.2%: 格式化为百分比，保留2位小数
        logger.info(f"置信度: {result['confidence']:.2%}")
        logger.info("-" * 50)  # 分隔线
    correct_rate = correct / total
    logger.info(f"第{i}轮准确率: {correct}/{total} = {correct_rate:.2%}")
    total_acuuracy.append(correct_rate)
    start += size
logger.info("=" * 50)  # 分隔线
logger.info("批量预测完成")
logger.info("各轮准确率汇总：")
for j in range(len(total_acuuracy)):
    logger.info(f"第{j}轮准确率: {total_acuuracy[j]:.2%}")
average_accuracy = sum(total_acuuracy) / len(total_acuuracy)
logger.info(f"平均准确率: {average_accuracy:.2%}")
