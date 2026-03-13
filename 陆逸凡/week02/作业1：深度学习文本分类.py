import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tabulate import tabulate

'''
作业：调整 09_深度学习文本分类.py代码中模型的层数和节点个数，对比模型的loss变化。
思路是每一层都有不同的hidden size，一般hidden size的层数在64 128和256中间
添加三个节点，每一个节点包括了一个线性层，激活层还有一个droupout层
最后的结果通过表格或者excel的形式表现出来
'''

# 读取数据集，使用制表符分隔，没有表头
# 你如果要跑的话，请修改为你自己的路径
dataset = pd.read_csv("D:/AI/AI work/Week 1/第1周：课程介绍与大模型基础/Week01/dataset.csv", sep="\t", header=None)
# 提取第一列作为文本数据
texts = dataset[0].tolist()
# 提取第二列作为标签数据（字符串格式）
string_labels = dataset[1].tolist()
# 设置最大序列长度
max_len = 40
# 创建标签到索引的映射字典
# set(string_labels) 获取所有不重复的标签
# enumerate 为每个标签分配一个唯一的数字索引
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
# 将字符串标签转换为数字标签
numerical_labels = [label_to_index[label] for label in string_labels]

# 创建字符到索引的映射字典，初始化包含填充符<pad>，索引为0
char_to_index = {'<pad>': 0}
# 遍历所有文本，构建字符词汇表
for text in texts:
    for char in text:
        # 如果字符不在字典中，添加它并分配新索引
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

# 创建索引到字符的反向映射字典
index_to_char = {i: char for char, i in char_to_index.items()}


# 定义自定义数据集类
class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        # 初始化文本数据
        self.texts = texts
        # 将标签转换为PyTorch张量
        self.labels = torch.tensor(labels, dtype=torch.long)
        # 字符到索引的映射
        self.char_to_index = char_to_index
        # 最大序列长度
        self.max_len = max_len
        # 词汇表大小
        self.vocab_size = vocab_size
        # 创建词袋向量表示
        self.bow_vectors = self._create_bow_vectors()

    # 创建词袋向量表示的方法
    def _create_bow_vectors(self):
        tokenized_texts = []
        # 遍历所有文本
        for text in self.texts:
            # 将字符转换为索引，限制长度不超过max_len
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            # 如果长度不足max_len，用0填充（<pad>）
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        bow_vectors = []
        # 为每个tokenized文本创建词袋向量
        for text_indices in tokenized_texts:
            # 初始化一个全零向量，长度为词汇表大小
            bow_vector = torch.zeros(self.vocab_size)
            # 遍历文本中的每个字符索引
            for index in text_indices:
                # 如果不是填充符（索引不为0）
                if index != 0:
                    # 在对应位置计数加1
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        # 将所有词袋向量堆叠成一个张量
        return torch.stack(bow_vectors)

    # 返回数据集大小
    def __len__(self):
        return len(self.texts)

    # 根据索引获取数据样本
    def __getitem__(self, idx):
        # 返回词袋向量和对应的标签
        return self.bow_vectors[idx], self.labels[idx]

# 定义简单的分类器模型
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        '''
        这里就改成以循环的方式添加
        这里的hidden_dim因为是列表，需要循环遍历，一个节点一个节点过
        一个节点包括了线性层，激活函数还有droupout
        最后再加一个全连接线性层输出
        '''
        # 调用父类初始化
        super(SimpleClassifier, self).__init__()
        
        # 记录前一层的大小，初始为输入维度
        pre_size = input_dim
        layers = []
        
        # 循环创建隐藏层
        for hidden_dim in hidden_dims: 
            # 添加线性层：将前一层输出转换为当前隐藏层维度
            layers.append(nn.Linear(pre_size, hidden_dim))
            # 添加ReLU激活函数：引入非线性
            layers.append(nn.ReLU())
            # 添加Dropout层：防止过拟合，0.2表示丢弃20%的神经元
            layers.append(nn.Dropout(0.2))
            # 更新pre_size为当前层输出维度，作为下一层的输入维度
            pre_size = hidden_dim
        
        # 添加输出层：将最后一个隐藏层输出转换为类别数
        layers.append(nn.Linear(pre_size, output_dim))
        # 正确写法：使用星号解包列表，构建Sequential模型
        self.network = nn.Sequential(*layers)
        
    # 前向传播过程
    def forward(self, x):
        # 直接调用network进行前向传播
        return self.network(x)


# 获取词汇表大小
vocab_size = len(char_to_index)
# 创建数据集实例
# vocab_size是词汇表大小，影响输入维度
# hidden_dim是隐藏层维度，会影响模型的表示能力和训练效果
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)

# 创建数据加载器，批量大小32，shuffle=True是打乱数据顺序
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

# 定义训练函数
def train_data(model_name, model, optimizer, criterion=nn.CrossEntropyLoss()):
    # 大的列表保存每轮的loss，第一个元素是模型名称
    epoch_losses = [model_name]
    print(f"{model_name}开始训练")
    
    # 训练轮数
    num_epochs = 10
    
    # 开始训练循环
    for epoch in range(num_epochs):
        # 设置模型为训练模式
        model.train()
        # 初始化累计损失
        running_loss = 0.0

        # 遍历数据加载器中的每个批次
        for idx, (inputs, labels) in enumerate(dataloader):
            # 清空优化器中的梯度（防止梯度累积）
            optimizer.zero_grad()
            # 前向传播：获取模型预测
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, labels)
            # 反向传播：计算梯度
            loss.backward()
            # 更新参数
            optimizer.step()
            # 累加损失
            running_loss += loss.item()
        
        # 计算该轮的平均损失
        avg_epoch_loss = running_loss / len(dataloader)
        # 将平均损失保留3位小数并添加到列表
        epoch_losses.append(round(avg_epoch_loss, 3))
        # 打印每个epoch的平均损失
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")
    
    # 返回损失列表和训练好的模型
    return epoch_losses, model

# 定义文本分类函数
def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    # 将文本转换为索引序列
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    # 填充不足长度的部分
    tokenized += [0] * (max_len - len(tokenized))

    # 创建词袋向量
    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    # 添加批次维度（将1D向量变为2D张量，batch_size=1）
    bow_vector = bow_vector.unsqueeze(0)

    # 设置模型为评估模式
    model.eval()
    # 禁用梯度计算以节省内存和计算资源
    with torch.no_grad():
        # 前向传播获取预测
        output = model(bow_vector)
    
    # 获取最大值的索引（预测类别）
    # dim=1表示在类别维度上找最大值
    _, predicted_index = torch.max(output, 1)
    # 将张量转换为Python标量
    predicted_index = predicted_index.item()
    # 将索引转换回标签
    predicted_label = index_to_label[predicted_index]

    return predicted_label

# 创建索引到标签的反向映射
index_to_label = {i: label for label, i in label_to_index.items()}

if __name__ == "__main__":
    # 输出维度等于类别数量
    output_dim = len(label_to_index)
    
    # 获取词汇表大小，这个就是输入的维度了
    input_dim = len(char_to_index)

    # 定义模型配置字典，键为模型名称，值为隐藏层维度列表
    model_configs = {
        "1层-32节点": [32],                      # 单层32个神经元
        "1层-64节点": [64],                      # 单层64个神经元
        "1层-128节点": [128],                    # 单层128个神经元
        "2层-64-32": [64, 32],                   # 两层：64->32
        "2层-128-64": [128, 64],                 # 两层：128->64
        "2层-256-128": [256, 128],               # 两层：256->128
        "3层-128-64-32": [128, 64, 32],          # 三层：128->64->32
        "3层-256-128-64": [256, 128, 64],        # 三层：256->128->64
        "4层-256-128-64-32": [256, 128, 64, 32], # 四层：256->128->64->32
        "5层-512-256-128-64-32": [512, 256, 128, 64, 32]  # 五层：512->256->128->64->32
    }
    
    # 测试数据列表
    test_datas = [
        "帮我导航到北京",
        "查询明天北京的天气",
        "播放周杰伦的歌",
        "今天天气怎么样",
        "设置明天早上七点的闹钟",
        "打开客厅的灯",
        "关闭空调"
    ]
    
    # 存储所有模型的训练损失
    total_loss_table = []
    # 存储所有模型的预测结果
    total_predict_table = []
    
    # 遍历所有模型配置
    for model_name, hidden_size_list in model_configs.items():
        # 创建模型实例
        model = SimpleClassifier(input_dim, hidden_size_list, output_dim)
        # 定义损失函数：交叉熵损失（内部包含softmax）
        criterion = nn.CrossEntropyLoss()
        # 定义优化器：Adam优化器，学习率0.001
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 训练模型，获取损失列表和训练好的模型
        epoch_losses, model = train_data(model_name, model, optimizer, criterion)
        total_loss_table.append(epoch_losses)
        
        # 创建预测结果行，第一列是模型名称
        p_table = []
        p_table.append(model_name)
        
        # 对每个测试文本进行预测
        for test_data in test_datas:
            predicted_class = classify_text(test_data, model, char_to_index, vocab_size, max_len, index_to_label)
            # 将预测结果添加到该行的列表中
            p_table.append(predicted_class)
        
        # 将该模型的预测结果添加到总表中
        total_predict_table.append(p_table)

    # 输出训练损失的表格
    headers_train = ["层名"]
    for i in range(10):
        headers_train.append(f"第{i+1}轮")
    print(tabulate(total_loss_table, headers=headers_train, tablefmt="grid"))
    
    # 输出预测结果的表格
    header_predict = ["层数"]
    for data in test_datas:
        header_predict.append(data)
    print(tabulate(total_predict_table, headers=header_predict, tablefmt="grid"))

    # 将loss数据保存为CSV文件
    excel_data = {}
    
    # 将列表数据转换为字典格式，便于创建DataFrame
    indexing = 0
    for loss in total_loss_table:
        for count, label in enumerate(headers_train):
            if indexing == 0:
                # 第一行：初始化字典
                excel_data[label] = [loss[count]]
            else:
                # 后续行：添加到已有列表
                excel_data[label].append(loss[count])
        indexing = indexing + 1
            
    df1 = pd.DataFrame(excel_data)
    # 保存到CSV文件，使用utf-8-sig编码支持中文
    # 你如果要跑的话，请修改为你自己的路径
    df1.to_csv('D:/AI/AI work/Week 2/Github homework/nlp_lyf/陆逸凡/week02/loss.csv', index=False, encoding='utf-8-sig')
    
    # 将predict数据保存为CSV文件
    predict_data = {}
    predict_index = 0
    
    # 将预测结果转换为字典格式
    for predicting in total_predict_table:
        for count, label in enumerate(header_predict):
            if predict_index == 0:
                predict_data[label] = [predicting[count]]
            else:
                predict_data[label].append(predicting[count])
        predict_index = predict_index + 1
    
    df2 = pd.DataFrame(predict_data)
    # 你如果要跑的话，请修改为你自己的路径
    df2.to_csv('D:/AI/AI work/Week 2/Github homework/nlp_lyf/陆逸凡/week02/predict.csv', index=False, encoding='utf-8-sig')