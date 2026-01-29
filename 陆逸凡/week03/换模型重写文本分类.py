# 导入必要的库
from matplotlib import colors  # 颜色管理
from tabulate import tabulate  # 表格化输出
import numpy as np  # 数值计算
import torch  # PyTorch深度学习框架
import torch.nn as nn  # PyTorch神经网络模块
from torch.utils.data import Dataset, DataLoader  # 数据集和数据加载器
import pandas as pd  # 数据处理
import torch.optim as optim  # 优化器
import matplotlib.pyplot as plt  # 绘图

# 第一步：数据预处理
# 读取CSV文件，使用制表符分隔，没有表头
dataset = pd.read_csv("D:/AI/AI work/dataset.csv", sep="\t", header=None)

# 将文本和标签分开，分别转换为列表
texts = dataset[0].tolist()  # 文本数据
labels = dataset[1].tolist()  # 对应的标签

# 将标签转换为数字索引
# 使用set去重，确保每个标签对应一个唯一的数字
label_to_index = {label: index for index, label in enumerate(set(labels))}
# 创建反向映射：数字索引 -> 原始标签
label_to_index_reverse = {index: label for label, index in label_to_index.items()}
# 将所有标签转换为对应的数字索引
numerical_labels = [label_to_index[label] for label in labels]

# 创建字符到索引的映射，用于将文本转换为数字序列
text_to_index = {'<pad>': 0}  # 添加填充字符的映射
for text in texts:
    for char in str(text):  # 遍历每个文本的每个字符
        if char not in text_to_index:  # 如果字符不在字典中
            text_to_index[char] = len(text_to_index)  # 添加新的映射
# 创建反向映射：索引 -> 字符
text_to_index_reverse = {i: char for i, char in enumerate(text_to_index)}

# 自定义数据集类，继承自PyTorch的Dataset
class CharDataset(Dataset):
    def __init__(self, texts, labels, text_to_index, max_len):
        """
        初始化数据集
        Args:
            texts: 文本列表
            labels: 标签列表
            text_to_index: 字符到索引的映射字典
            max_len: 文本最大长度，用于填充或截断
        """
        self.texts = texts  # 存储文本
        self.labels = torch.tensor(labels, dtype=torch.long)  # 转换为tensor
        self.text_to_index = text_to_index  # 字符索引映射
        self.max_len = max_len  # 最大序列长度

    def __len__(self):
        """返回数据集样本数量"""
        return len(self.texts)

    def __getitem__(self, idx):
        """获取单个样本"""
        text = self.texts[idx]  # 获取指定索引的文本
        # 将文本转换为索引序列，如果字符不存在则使用0（<pad>）
        indices = [self.text_to_index.get(char, 0) for char in text[:self.max_len]]
        # 填充：如果序列长度小于max_len，用0填充
        indices += [0] * (self.max_len - len(indices))
        # 返回文本索引序列和对应的标签
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

# 序列模型类，继承自nn.Module
class sequence_model(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, layer_type, num_layers=2, dropout=0.3):
        """
        初始化序列模型
        Args:
            vocab_size: 词汇表大小（输入维度）
            embedding_size: 词嵌入维度
            hidden_size: 隐藏层维度
            output_size: 输出类别数量
            layer_type: 序列层类型（RNN/LSTM/GRU）
            num_layers: RNN层数
            dropout: Dropout概率
        """
        super(sequence_model, self).__init__()  # 调用父类初始化
        
        # 词嵌入层：将索引映射为稠密向量
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        
        # 根据类型选择不同的序列层
        if layer_type == 'LSTM':
            # LSTM层：处理长期依赖关系
            self.act = nn.LSTM(embedding_size, hidden_size, batch_first=True, 
                              dropout=dropout if num_layers > 1 else 0)
        elif layer_type == 'GRU':
            # GRU层：LSTM的简化版，计算更快
            self.act = nn.GRU(embedding_size, hidden_size, batch_first=True,
                             dropout=dropout if num_layers > 1 else 0)
        elif layer_type == 'RNN':
            # 标准RNN层
            self.act = nn.RNN(embedding_size, hidden_size, batch_first=True,
                             dropout=dropout if num_layers > 1 else 0)
        
        # BatchNorm层：稳定训练，加速收敛
        # 作用：使每一层的输入分布保持稳定（均值为0，方差为1）
        #       防止梯度消失或梯度爆炸，允许使用更大的学习率
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # Dropout层：防止过拟合
        # 作用：通过随机丢弃一部分神经元的输出，减少模型对训练数据的过度依赖
        #       提高模型的泛化能力和鲁棒性
        self.dropout = nn.Dropout(dropout)
        
        # 全连接层：将隐藏状态映射到输出类别
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, layer_type):
        """
        前向传播
        Args:
            x: 输入文本索引序列
            layer_type: 序列层类型
        Returns:
            out: 模型输出（未归一化的类别分数）
        """
        # 将索引转换为词嵌入向量
        x = self.embedding(x)
        
        # 通过序列层（RNN/LSTM/GRU）
        if layer_type == 'LSTM':
            # LSTM返回输出和(hidden_state, cell_state)元组
            output, (h, c) = self.act(x)
        else:
            # RNN/GRU返回输出和hidden_state
            output, h = self.act(x)
        
        # h的形状是 (num_layers, batch_size, hidden_size)
        # 只需要最后一层的隐藏状态
        h = h[-1]  # 取最后一层，形状变为 (batch_size, hidden_size)
        
        # 应用BatchNorm：归一化隐藏状态，加速训练
        h = self.batch_norm(h)
        
        # 应用Dropout：随机丢弃部分神经元，防止过拟合
        h = self.dropout(h)
        
        # 通过全连接层，得到每个类别的分数
        out = self.fc(h)
        
        return out

# 定义损失函数：交叉熵损失，用于多分类问题
criterion = nn.CrossEntropyLoss()

def train(model, train_loader, optimizer, epoch, model_type):
    """
    训练函数
    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        epoch: 当前训练轮次
        model_type: 模型类型
    Returns:
        loss_mean: 本轮训练的平均损失
    """
    model.train()  # 设置为训练模式，启用Dropout和BatchNorm
    loss_total = 0.0  # 累计损失

    for batch_idx, (data, target) in enumerate(train_loader):
        # 前向传播：计算模型输出
        outputs = model(data, model_type)
        # 计算损失：预测输出与真实标签的差异
        loss = criterion(outputs, target)
        
        # 反向传播：计算梯度
        optimizer.zero_grad()  # 清空历史梯度
        loss.backward()  # 反向传播计算梯度
        
        # 梯度裁剪：防止梯度爆炸（对RNN特别重要）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()  # 更新模型参数
        
        # 计算本轮batch的损失贡献
        running_loss = loss.item() / len(train_loader)
        loss_total += running_loss  # 累加损失
    
    # 计算本轮平均损失
    loss_mean = loss_total / len(train_loader)
    print(f"第{epoch}轮的loss: {loss_mean:.4f}")
    return loss_mean

# 测试文本列表，用于评估模型效果
check_texts = [
    "明天上海下雨吗",      # 天气查询
    "导航去清华大学",      # 导航指令
    "播放周杰伦的歌",      # 音乐播放
    "今天股市怎么样",      # 股市查询
    "翻译英文到中文",      # 翻译功能
    "帮我设置闹钟",        # 闹钟设置
    "最近有什么新闻",      # 新闻查询
    "计算器打开一下",      # 计算器打开
    "微信发给张三消息",    # 消息发送
    "打开相机拍照"         # 相机功能
]

def check_result(model, max_len, model_type):
    """
    评估函数：在测试文本上评估模型效果
    Args:
        model: 训练好的模型
        max_len: 最大序列长度
        model_type: 模型类型
    Returns:
        testing_results: 测试结果列表
    """
    model.eval()  # 设置为评估模式，禁用Dropout和BatchNorm
    testing_results = [model_type]  # 结果列表，第一个元素是模型类型
    
    with torch.no_grad():  # 不计算梯度，节省内存和计算资源
        for check_text in check_texts:  # 遍历每个测试文本
            # 将测试文本转换为索引序列
            input_test_text = [text_to_index.get(char, 0) for char in check_text[:max_len]]
            # 填充序列到最大长度
            if len(input_test_text) < max_len:
                input_test_text += [0] * (max_len - len(input_test_text))
            
            # 添加batch维度（batch_size=1），转换为tensor
            input_tensor = torch.tensor(input_test_text, dtype=torch.long).unsqueeze(0)
            
            # 模型预测
            result = model(input_tensor, model_type)
            
            # 找到概率最高的类别
            # torch.max返回最大值和最大值的索引
            _, max_possibility = torch.max(result, 1)  # 在维度1（类别维度）上取最大值
            
            # 将数字索引转换回原始标签
            result_label = label_to_index_reverse[max_possibility.item()]
            testing_results.append(result_label)  # 添加到结果列表
    
    return testing_results

def main():
    """
    主函数：组织整个训练和评估流程
    """
    # 训练轮数
    epochs = 10
    # 词汇表大小
    input_size = len(text_to_index)
    # 词嵌入维度
    embedding_size = 64
    # 隐藏层维度
    hidden_size = 128
    # 输出类别数量
    output_size = len(label_to_index)
    # 最大序列长度
    max_len = 40
    
    # 要训练的模型类型列表
    model_types = ['LSTM', 'GRU', 'RNN']
    
    # 创建数据集和数据加载器
    train_loader = CharDataset(texts, numerical_labels, text_to_index, max_len)
    dataloader = DataLoader(train_loader, batch_size=64, shuffle=True)  # 每批64个样本，打乱数据
    
    # 存储每个模型的训练历史
    all_losses = {}  
    # 存储每个模型的测试结果
    all_testing_results = []
    
    # 遍历每种模型类型进行训练
    for model_type in model_types:
        print(f"现在开始{model_type}模型的训练")
        
        # 创建模型实例
        model = sequence_model(input_size, embedding_size, hidden_size, output_size, layer_type=model_type)
        loss_history = []  # 当前模型的损失历史
        
        # 使用Adam优化器，学习率0.001
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 训练循环
        for epoch in range(epochs):
            # 训练一个epoch，获取平均损失
            loss = train(model, dataloader, optimizer, epoch, model_type)
            loss_history.append(loss)  # 记录损失
        
        # 存储当前模型的损失历史
        all_losses[model_type] = loss_history
        # print(f"{model_type}损失历史: {loss_history}")
        
        # 在测试集上评估模型
        test_result = check_result(model, max_len, model_type)
        all_testing_results.append(test_result)  # 存储测试结果
    
    # 打印测试结果表格
    header = ["模型名称"] + check_texts  # 表头：模型名称+各个测试文本
    print(tabulate(all_testing_results, headers=header, tablefmt="grid"))
    
    # 绘制损失曲线图
    colors = ['b', 'g', 'r']  # 为不同模型设置不同颜色
    plt.figure(figsize=(10, 6))  # 设置图形大小
    for idx, model_type in enumerate(model_types):
        if model_type in all_losses:
            # 绘制每个模型的损失曲线
            plt.plot(range(1, epochs+1), all_losses[model_type], 
                    label=f'{model_type}', color=colors[idx], marker='o')
    
    # 设置图表属性
    plt.title('Training Loss Comparison')  # 标题
    plt.xlabel('Epochs')  # X轴标签
    plt.ylabel('Loss')  # Y轴标签
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格
    plt.show()  # 显示图形

# 程序入口
if __name__ == '__main__':
    main()