import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

# 标签：字符串→数字索引
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]
index_to_label = {i: label for label, i in label_to_index.items()}
output_dim = len(label_to_index)  # 分类任务输出维度=标签类别数

# 字符：构建字符表（含<pad>填充符，索引0）
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)
vocab_size = len(char_to_index)  # 字符表大小
max_len = 40  # 文本最大长度（截断/填充）

train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, numerical_labels, test_size=0.2, random_state=42, stratify=numerical_labels
)

class CharRNNDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # 截断+填充：统一序列长度为max_len
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]  # 截断
        indices += [0] * (self.max_len - len(indices))  # 填充（<pad>对应索引0）
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

# 构建训练/测试数据加载器
train_dataset = CharRNNDataset(train_texts, train_labels, char_to_index, max_len)
test_dataset = CharRNNDataset(test_texts, test_labels, char_to_index, max_len)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# LSTM
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # 字符嵌入层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # LSTM层（batch_first=True）
        self.fc = nn.Linear(hidden_dim, output_dim)  # 分类头（全连接层）

    def forward(self, x):
        embedded = self.embedding(x)  # [batch, seq_len] → [batch, seq_len, emb_dim]
        lstm_out, (hidden, cell) = self.lstm(embedded)  # hidden: [1, batch, hidden_dim]
        out = self.fc(hidden.squeeze(0))  # 挤压num_layers维度 → [batch, hidden_dim] → [batch, output_dim]
        return out

# RNN
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)  # [batch, seq_len] → [batch, seq_len, emb_dim]
        rnn_out, hidden = self.rnn(embedded)  # RNN仅返回输出和最终隐藏状态（无细胞状态）
        out = self.fc(hidden.squeeze(0))  # 挤压维度 → [batch, output_dim]
        return out

# GRU
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)  # [batch, seq_len] → [batch, seq_len, emb_dim]
        gru_out, hidden = self.gru(embedded)
        out = self.fc(hidden.squeeze(0))  # 挤压维度 → [batch, output_dim]
        return out

def train_model(model, train_loader, criterion, optimizer, num_epochs=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()  # 训练模式（开启Dropout/BatchNorm等）
    for epoch in range(num_epochs):
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # 梯度清零
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            running_loss += loss.item()
            # 每50个批次打印一次损失
            if idx % 50 == 0 and idx > 0:
                print(f"  Batch {idx}, Avg Loss: {running_loss/idx:.4f}")
        # 每轮打印平均损失
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}")
    return model

def test_model(model, test_loader):
    """
    模型测试函数，计算测试集精度
    :param model: 训练完成的模型
    :param test_loader: 测试数据加载器
    :return: 测试集精度（accuracy）
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()  # 评估模式（关闭Dropout/BatchNorm等）
    all_preds = []
    all_labels = []
    with torch.no_grad():  # 关闭梯度计算，节省内存
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)  # 取logits最大值对应的索引为预测标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    # 计算精度
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

# 超参数（3类模型完全一致）
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
NUM_EPOCHS = 4
LR = 0.001

# 存储模型名称和对应精度
model_metrics = {}

print("="*50)
print("开始训练LSTM模型...")
lstm_model = LSTMClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, output_dim)
lstm_criterion = nn.CrossEntropyLoss()
lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=LR)
lstm_model = train_model(lstm_model, train_loader, lstm_criterion, lstm_optimizer, NUM_EPOCHS)
lstm_acc = test_model(lstm_model, test_loader)
model_metrics["LSTM"] = lstm_acc
print(f"LSTM模型测试精度: {lstm_acc:.4f}")

print("="*50)
print("开始训练RNN模型...")
rnn_model = RNNClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, output_dim)
rnn_criterion = nn.CrossEntropyLoss()
rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=LR)
rnn_model = train_model(rnn_model, train_loader, rnn_criterion, rnn_optimizer, NUM_EPOCHS)
rnn_acc = test_model(rnn_model, test_loader)
model_metrics["RNN"] = rnn_acc
print(f"RNN模型测试精度: {rnn_acc:.4f}")

print("="*50)
print("开始训练GRU模型...")
gru_model = GRUClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, output_dim)
gru_criterion = nn.CrossEntropyLoss()
gru_optimizer = optim.Adam(gru_model.parameters(), lr=LR)
gru_model = train_model(gru_model, train_loader, gru_criterion, gru_optimizer, NUM_EPOCHS)
gru_acc = test_model(gru_model, test_loader)
model_metrics["GRU"] = gru_acc
print(f"GRU模型测试精度: {gru_acc:.4f}")

print("="*50)
print("【3类模型测试精度对比】")
for model_name, acc in sorted(model_metrics.items(), key=lambda x: x[1], reverse=True):
    print(f"{model_name}: {acc:.4f}")

def classify_text(text, model, char_to_index, max_len, index_to_label):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    # 文本预处理：截断+填充
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)  # 增加batch维度
    # 预测
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_idx = torch.max(output, 1)
    return index_to_label[predicted_idx.item()]

# 测试预测
print("="*50)
print("【自定义文本预测示例】")
test_texts = ["帮我导航到北京", "查询明天北京的天气", "播放周杰伦的七里香"]
for text in test_texts:
    lstm_pred = classify_text(text, lstm_model, char_to_index, max_len, index_to_label)
    rnn_pred = classify_text(text, rnn_model, char_to_index, max_len, index_to_label)
    gru_pred = classify_text(text, gru_model, char_to_index, max_len, index_to_label)
    print(f"输入: {text}")
    print(f"  LSTM预测: {lstm_pred}, RNN预测: {rnn_pred}, GRU预测: {gru_pred}")
    print("-"*30)
