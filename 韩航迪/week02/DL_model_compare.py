import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt  # 新增：用于loss可视化

# 读取数据
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

# 标签转数字
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

# 字符转索引（含<pad>）
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

class ConfigurableClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        """
        可配置层数和节点数的分类器
        :param input_dim: 输入维度（词汇表大小）
        :param hidden_dims: 隐藏层维度列表，如[64]（1层64节点）、[128, 64]（2层，128→64）
        :param output_dim: 输出维度（标签数）
        """
        super(ConfigurableClassifier, self).__init__()
        self.layers = nn.ModuleList()  # 存储所有层
        # 输入层 → 第一层隐藏层
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        # 新增：多层隐藏层（支持任意层数）
        for i in range(1, len(hidden_dims)):
            self.layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
        # 最后一层隐藏层 → 输出层
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.relu = nn.ReLU()  # 激活函数

    def forward(self, x):
        out = x
        # 遍历所有层（除输出层外，每层后加ReLU）
        for layer in self.layers[:-1]:
            out = layer(out)
            out = self.relu(out)
        # 输出层不加激活（CrossEntropyLoss自带softmax）
        out = self.layers[-1](out)
        return out

def train_model(hidden_dims, num_epochs=10, batch_size=32):
    """
    训练模型并返回每轮的loss
    :param hidden_dims: 隐藏层维度列表
    :return: 每轮epoch的平均loss列表
    """
    # 初始化数据集和dataloader
    char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
    dataloader = DataLoader(char_dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型、损失函数、优化器
    output_dim = len(label_to_index)
    model = ConfigurableClassifier(vocab_size, hidden_dims, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 记录每轮的loss
    epoch_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 计算本轮平均loss
        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"隐藏层配置 {hidden_dims} | Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return epoch_losses

# 实验1：原配置（1层，128节点）
loss_1layer_128 = train_model(hidden_dims=[128], num_epochs=10)

# 实验2：1层，64节点（减少节点数）
loss_1layer_64 = train_model(hidden_dims=[64], num_epochs=10)

# 实验3：1层，256节点（增加节点数）
loss_1layer_256 = train_model(hidden_dims=[256], num_epochs=10)

# 实验4：2层，[128, 64]（增加层数，节点数递减）
loss_2layer_128_64 = train_model(hidden_dims=[128, 64], num_epochs=10)


plt.figure(figsize=(10, 6))
plt.plot(loss_1layer_64, label="1层-64节点", marker='o')
plt.plot(loss_1layer_128, label="1层-128节点（原配置）", marker='s')
plt.plot(loss_1layer_256, label="1层-256节点", marker='^')
plt.plot(loss_2layer_128_64, label="2层-128→64节点", marker='*')

plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.title("不同模型配置的Loss变化对比")
plt.legend()
plt.grid(True)
plt.savefig("loss_comparison.png")  # 保存图片
plt.show()


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


# 验证最优模型（示例：2层配置）
index_to_label = {i: label for label, i in label_to_index.items()}
best_model = ConfigurableClassifier(vocab_size, [128, 64], len(label_to_index))

new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, best_model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, best_model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
