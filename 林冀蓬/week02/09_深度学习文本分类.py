import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

dataset = pd.read_csv("../week01/dataset.csv", sep="\t", header=None)
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
            tokenized += [0] * (max_len - len(tokenized))
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


# 创建一个灵活的模型类，支持不同层数和节点数
class FlexibleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        # hidden_dims是一个列表，每个元素表示一层的节点数
        super(FlexibleClassifier, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        # 创建隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        
        # 创建输出层
        layers.append(nn.Linear(current_dim, output_dim))
        
        # 将所有层组合成序列
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# 准备数据
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

# 定义不同的模型结构配置
model_configs = [
    {"name": "2层-64节点", "hidden_dims": [64]},
    {"name": "2层-128节点", "hidden_dims": [128]},
    {"name": "2层-256节点", "hidden_dims": [256]},
    {"name": "3层-64-64节点", "hidden_dims": [64, 64]},
    {"name": "3层-128-64节点", "hidden_dims": [128, 64]},
    {"name": "4层-128-64-32节点", "hidden_dims": [128, 64, 32]}
]

output_dim = len(label_to_index)
num_epochs = 10

# 存储所有模型的loss历史
all_loss_histories = {}

# 训练每个模型
for config in model_configs:
    print(f"\n=== 训练模型: {config['name']} ===")
    
    # 创建模型
    model = FlexibleClassifier(vocab_size, config['hidden_dims'], output_dim)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # 记录loss历史
    loss_history = []
    
    # 训练模型
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
        
        epoch_loss = running_loss / len(dataloader)
        loss_history.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    # 保存loss历史
    all_loss_histories[config['name']] = loss_history

# 可视化对比不同模型的loss变化
plt.figure(figsize=(12, 8))
for model_name, loss_history in all_loss_histories.items():
    plt.plot(range(1, num_epochs + 1), loss_history, label=model_name)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('不同模型结构的Loss变化对比')
plt.legend()
plt.grid(True)
plt.savefig('model_loss_comparison.png')
plt.show()

# 打印最终loss对比
print("\n=== 最终Loss对比 ===")
for model_name, loss_history in all_loss_histories.items():
    print(f"{model_name}: {loss_history[-1]:.4f}")

# 测试最佳模型（以2层-128节点为例）
print("\n=== 测试模型性能 ===")
best_model = FlexibleClassifier(vocab_size, [128], output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(best_model.parameters(), lr=0.01)

# 重新训练最佳模型
for epoch in range(num_epochs):
    best_model.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = best_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

# 测试示例文本
index_to_label = {i: label for label, i in label_to_index.items()}

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

# 测试文本
new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, best_model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, best_model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")