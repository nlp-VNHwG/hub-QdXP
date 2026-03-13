作业1：
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

max_len = 40


class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        tokenized_texts = []
        for text in self.texts:
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        bow_vectors = []
        for text_indices in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size)
            for index in text_indices:
                if index != 0:
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(SimpleClassifier, self).__init__()

        # 允许能传256，128
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        # 维度链：input -> h1 -> h2 -> ... -> output
        dims = [input_dim] + hidden_dims + [output_dim]

        # 用 ModuleList 装多层 Linear
        self.fcs = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.fcs.append(nn.Linear(dims[i], dims[i + 1]))

        self.relu = nn.ReLU()

    def forward(self, x):
        out = x

        # 前面所有隐藏层：Linear + ReLU
        for i in range(len(self.fcs) - 1):
            out = self.fcs[i](out)
            out = self.relu(out)
        out = self.fcs[-1](out)
        return out


char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size) # 读取单个样本
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True) # 读取批量数据集 -》 batch数据

hidden_dim = [128,64]
output_dim = len(label_to_index)
model = SimpleClassifier(vocab_size, hidden_dim, output_dim) # 维度和精度有什么关系？
criterion = nn.CrossEntropyLoss() # 损失函数 内部自带激活函数，softmax
optimizer = optim.Adam(model.parameters(), lr=0.01)

# epoch： 将数据集整体迭代训练一次
# batch： 数据集汇总为一批训练一次

num_epochs = 10
for epoch in range(num_epochs): # 12000， batch size 100 -》 batch 个数： 12000 / 100
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
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")


    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")


def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))

    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    bow_vector = bow_vector.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(bow_vector)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label


index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")



作业2：
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
#1、准备数据
X_numpy = np.random.rand(100,1) * 10
y_numpy = np.sin(X_numpy) + 1 + 0.1 *np.random.randn(100,1)

#2、格式转化
X = torch.from_numpy(X_numpy).float()
Y = torch.from_numpy(y_numpy).float()

#3、定义模型
class MLP(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(MLP, self).__init__()
        if isinstance(hidden_dim,int):
            hidden_dim = [hidden_dim]

        dims = [input_dim] + hidden_dim + [output_dim]

        self.fcs = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.fcs.append(nn.Linear(dims[i], dims[i+1]))#所有网络层都连接起来,但是这些都是线性的

        self.relu = nn.ReLU()

    def forward(self,x):
        out = x
        for i in range(len(self.fcs) - 1):
            out = self.relu(self.fcs[i](out))

        out =self.fcs[-1](out)
        return out

#4、构建即将要训练的模型
hidden_dim = [128,56]
model = MLP(input_dim = 1,hidden_dim = hidden_dim, output_dim = 1)
loss_function = torch.nn.MSELoss()#损失函数 线性回归一般都用均方差
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)#对于复杂非凸优化问题 Adam比SGD更加适用

#5、训练模型
num_epochs = 5000
for epoch in range(num_epochs):
    model.train()
    y_predicted = model(X)
    loss = loss_function(y_predicted, Y)

    optimizer.zero_grad()#梯度归0操作
    loss.backward()#
    optimizer.step()

    if (epoch + 1) % 100 == 0:#每500打印一次loss
        print(f'epoch [{epoch + 1}/{num_epochs}], loss {loss.item():.4f}')

#6、对结果作可视化处理
#6.1 获取对应x的区间
x_min = float(X.min())
x_max = float(X.max())
#6.2获取更密集的数组去作为图形绘制的输入,同时要注意格式 torch默认的格式是[batch_size,input_dim]
X_line = np.linspace(x_min, x_max, 1000).reshape(-1, 1)
X_line_t = torch.from_numpy(X_line).float()
#6.3获取模型预测得到的
with torch.no_grad():
    y_line_t =model(X_line_t)
    y_line = y_line_t.numpy()

plt.figure(figsize = (10,10))
plt.scatter(X_numpy, y_numpy, label = "data")#绘制原始数据点
plt.plot(X_line, y_line, label = "model fit")
plt.xlabel("x")
plt.ylabel("y")
plt.title("MLP fit: y = sinx + 0.5 + noise")
plt.legend()
plt.grid(True)
plt.show()
