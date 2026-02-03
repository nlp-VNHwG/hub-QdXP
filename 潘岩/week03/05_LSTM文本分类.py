import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score  # 用于计算预测精度

# ===================== 配置项：切换模型 =====================
MODEL_TYPE = "GRU"  # 可选：RNN / LSTM / GRU
# ===================== 固定超参数 =====================
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 4
MAX_LEN = 40
DATA_PATH = "../Week01/dataset.csv"  

# ===================== 1. 数据加载与预处理 =====================
# 读取数据
dataset = pd.read_csv(DATA_PATH, sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

# 标签转数字
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]
index_to_label = {i: label for label, i in label_to_index.items()}
output_dim = len(label_to_index)

# 字符转索引（构建字符表）
char_to_index = {'<pad>': 0}  # <pad>对应索引0，用于填充
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)
vocab_size = len(char_to_index)

# ===================== 2. 自定义数据集类（与原始代码一致） =====================
class CharDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # 字符转索引，截断/填充到固定长度
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

# 构建数据集和数据加载器
dataset = CharDataset(texts, numerical_labels, char_to_index, MAX_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ===================== 3. 通用循环分类模型（支持RNN/LSTM/GRU） =====================
class RecurrentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, model_type="LSTM"):
        super(RecurrentClassifier, self).__init__()
        self.model_type = model_type
        self.hidden_dim = hidden_dim

        # Embedding层（与原始代码一致，可训练）
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)  # 新增padding_idx=0，屏蔽填充位
        
        # 循环层：根据model_type选择RNN/LSTM/GRU
        if self.model_type == "RNN":
            self.recurrent = nn.RNN(embedding_dim, hidden_dim, batch_first=True, bidirectional=False)
        elif self.model_type == "LSTM":
            self.recurrent = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=False)
        elif self.model_type == "GRU":
            self.recurrent = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=False)
        else:
            raise ValueError("model_type仅支持：RNN / LSTM / GRU")
        
        # 全连接层：将隐藏层输出映射到标签数
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # 循环层前向传播，处理不同返回值
        if self.model_type == "RNN" or self.model_type == "GRU":
            # RNN/GRU返回：输出序列 + 最后一步隐藏状态
            rec_out, hidden = self.recurrent(embedded)  # hidden: [1, batch_size, hidden_dim]
        else:  # LSTM
            # LSTM返回：输出序列 + (最后一步隐藏状态, 最后一步细胞状态)
            rec_out, (hidden, cell) = self.recurrent(embedded)  # hidden: [1, batch_size, hidden_dim]
        
        # 取最后一步隐藏状态做分类（挤压维度：[1, batch_size, hidden_dim] → [batch_size, hidden_dim]）
        out = self.fc(hidden.squeeze(0))  # [batch_size, output_dim]
        return out

# ===================== 4. 模型初始化与训练配置 =====================
# 初始化模型
model = RecurrentClassifier(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=output_dim,
    model_type=MODEL_TYPE
)

# 损失函数（分类任务用交叉熵）、优化器（Adam）
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 设备配置（自动使用GPU，无GPU则用CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"使用设备：{device} | 模型类型：{MODEL_TYPE}")

# ===================== 5. 训练函数（含损失打印+精度计算） =====================
def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    train_loss_history = []  # 记录每轮损失
    train_acc_history = []   # 记录每轮精度
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for idx, (inputs, labels) in enumerate(dataloader):
            # 数据移到设备上
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 前向传播
            optimizer.zero_grad()  # 清空梯度
            outputs = model(inputs)  # 模型输出：[batch_size, output_dim]
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 反向传播+优化
            loss.backward()
            optimizer.step()
            
            # 累计损失
            running_loss += loss.item()
            
            # 记录预测结果和真实标签（用于计算精度）
            _, preds = torch.max(outputs, 1)  # 取概率最大的索引为预测标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 打印批次信息
            if idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] | Batch [{idx}] | Batch Loss: {loss.item():.4f}")
        
        # 计算本轮平均损失和精度
        avg_loss = running_loss / len(dataloader)
        acc = accuracy_score(all_labels, all_preds)
        train_loss_history.append(avg_loss)
        train_acc_history.append(acc)
        
        # 打印本轮结果
        print("-" * 50)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Avg Loss: {avg_loss:.4f} | Train Accuracy: {acc:.4f}")
        print("-" * 50)
    
    return train_loss_history, train_acc_history

# ===================== 6. 预测函数（与原始代码一致，适配新模型） =====================
def classify_text(text, model, char_to_index, max_len, index_to_label, device):
    # 文本预处理：字符转索引+填充/截断
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)  # 增加batch维度并移到设备
    
    # 模型推理（关闭梯度）
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    
    # 取预测结果
    _, predicted_idx = torch.max(output, 1)
    predicted_label = index_to_label[predicted_idx.item()]
    return predicted_label

# ===================== 7. 开始训练 =====================
print(f"\n========== 开始训练 {MODEL_TYPE} 模型 ==========")
loss_hist, acc_hist = train_model(model, dataloader, criterion, optimizer, NUM_EPOCHS, device)

# ===================== 8. 测试预测 =====================
print(f"\n========== {MODEL_TYPE} 模型预测测试 ==========")
test_texts = [
    "帮我导航到北京",
    "查询明天北京的天气",
    "今天想吃火锅",
    "推荐一部好看的电影"
]
for text in test_texts:
    pred_label = classify_text(text, model, char_to_index, MAX_LEN, index_to_label, device)
    print(f"输入：{text} → 预测标签：{pred_label}")

# 打印最终训练结果
print(f"\n========== {MODEL_TYPE} 模型训练结果 ==========")
print(f"最后一轮损失：{loss_hist[-1]:.4f} | 最后一轮精度：{acc_hist[-1]:.4f}")
