import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- 1. 数据准备 (保持不变) ---
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

vocab_size = len(char_to_index)
max_len = 40


# --- 2. Dataset 定义 (保持不变) ---
class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        bow_vectors = []
        for text in self.texts:
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))
            bow_vector = torch.zeros(self.vocab_size)
            for index in tokenized:
                if index != 0: bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]


# --- 3. 核心修改：升级模型结构 ---
# 作业要求：调整层数和节点个数
class EnhancedClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EnhancedClassifier, self).__init__()
        # 第一层：输入层 -> 隐藏层1
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()

        # 第二层：增加的隐藏层2 (节点数减半，这是常用的漏斗结构)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu2 = nn.ReLU()

        # 第三层：隐藏层2 -> 输出层
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


# --- 4. 训练与对比 ---
HIDDEN_DIM = 256
BATCH_SIZE = 32
LR = 0.01

char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = EnhancedClassifier(vocab_size, HIDDEN_DIM, len(label_to_index))
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=LR)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"开始训练：隐藏层节点={HIDDEN_DIM}, 层数=3")
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/10], Loss: {avg_loss:.4f}")