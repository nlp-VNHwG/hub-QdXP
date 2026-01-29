import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ===================== 1. 数据预处理（复用原逻辑） =====================
dataset = pd.read_csv("../dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

# 标签映射
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]
index_to_label = {i: label for label, i in label_to_index.items()}

# 字符映射
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)
vocab_size = len(char_to_index)
max_len = 40

# 自定义数据集
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
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

# 构建数据集和DataLoader
dataset = CharRNNDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ===================== 2. 通用循环网络分类器 =====================
class RecurrentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, rnn_type="lstm"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 选择RNN/LSTM/GRU
        if rnn_type == "rnn":
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        elif rnn_type == "gru":
            self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)  # [batch, seq_len, embedding_dim]
        rnn_out, hidden = self.rnn(embedded)
        # 提取最后一层最后时间步的隐藏态（RNN/GRU: hidden; LSTM: hidden[0]）
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        hidden = hidden.squeeze(0)
        out = self.fc(hidden)
        return out

# ===================== 3. 训练与评估函数 =====================
def train_and_evaluate(rnn_type, epochs=4):
    """训练模型并返回最终精度"""
    # 初始化模型
    embedding_dim = 64
    hidden_dim = 128
    output_dim = len(label_to_index)
    model = RecurrentClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, rnn_type)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if idx % 50 == 0:
                print(f"[{rnn_type}] Epoch {epoch+1}, Batch {idx}, Loss: {loss.item():.4f}")
        avg_loss = running_loss / len(dataloader)
        print(f"[{rnn_type}] Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

    # 评估精度
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f"[{rnn_type}] 最终分类精度: {accuracy:.4f}\n" + "-"*50)
    return accuracy

# ===================== 4. 对比实验 =====================
if __name__ == "__main__":
    # 分别训练RNN/LSTM/GRU
    rnn_acc = train_and_evaluate("rnn")
    lstm_acc = train_and_evaluate("lstm")
    gru_acc = train_and_evaluate("gru")

    # 打印对比结果
    print("=== 精度对比 ===")
    print(f"RNN 精度: {rnn_acc:.4f}")
    print(f"LSTM 精度: {lstm_acc:.4f}")
    print(f"GRU 精度: {gru_acc:.4f}")