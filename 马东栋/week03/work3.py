import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time

dataset = pd.read_csv("E:\\BaiduNetdiskDownload\\nlp\\Week01\\Week01\\dataset.csv", sep="\t", header=None)
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
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

# RNN模型
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, hidden_state = self.rnn(embedded)
        out = self.fc(hidden_state.squeeze(0))
        return out

# LSTM模型
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)
        out = self.fc(hidden_state.squeeze(0))
        return out

# GRU模型
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, hidden_state = self.gru(embedded)
        out = self.fc(hidden_state.squeeze(0))
        return out

def train_model(model, dataloader, criterion, optimizer, num_epochs=4):
    model.train()
    start_time = time.time()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    training_time = time.time() - start_time
    return training_time

def evaluate_model_accuracy(model, test_texts, test_labels, char_to_index, max_len, index_to_label):
    model.eval()
    correct = 0
    total = len(test_texts)

    with torch.no_grad():
        for i, text in enumerate(test_texts):
            indices = [char_to_index.get(char, 0) for char in text[:max_len]]
            indices += [0] * (max_len - len(indices))
            input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

            output = model(input_tensor)
            _, predicted_index = torch.max(output, 1)
            predicted_label_idx = predicted_index.item()

            true_label_idx = test_labels[i]
            if predicted_label_idx == true_label_idx:
                correct += 1

    accuracy = correct / total
    return accuracy

def classify_text(text, model, char_to_index, max_len, index_to_label):
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label

# 准备数据
dataset_obj = CharDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(dataset_obj, batch_size=32, shuffle=True)

# 模型参数
embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)
index_to_label = {i: label for label, i in label_to_index.items()}

# 分割数据集用于评估（这里简单地用原数据集的一部分做演示）
split_ratio = 0.8
split_idx = int(len(texts) * split_ratio)
train_texts, test_texts = texts[:split_idx], texts[split_idx:]
train_labels, test_labels = numerical_labels[:split_idx], numerical_labels[split_idx:]

# 测试三种模型
models = {
    'RNN': RNNClassifier,
    'LSTM': LSTMClassifier,
    'GRU': GRUClassifier
}

results = {}

for name, ModelClass in models.items():
    print(f"\n=== 训练 {name} 模型 ===")

    # 创建模型实例
    model = ModelClass(vocab_size, embedding_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    training_time = train_model(model, dataloader, criterion, optimizer, num_epochs=4)

    # 评估准确率
    accuracy = evaluate_model_accuracy(model, test_texts, test_labels, char_to_index, max_len, index_to_label)

    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'training_time': training_time
    }

    print(f"{name} 模型:")
    print(f"  - 训练时间: {training_time:.2f}秒")
    print(f"  - 准确率: {accuracy:.4f}")

# 比较结果
print("\n=== 模型比较结果 ===")
for name, result in results.items():
    print(f"{name}: 准确率={result['accuracy']:.4f}, 训练时间={result['training_time']:.2f}秒")

# 示例预测
test_sentences = ["帮我导航到北京", "查询明天北京的天气"]
for sentence in test_sentences:
    print(f"\n测试句子: '{sentence}'")
    for name, result in results.items():
        predicted_class = classify_text(sentence, result['model'], char_to_index, max_len, index_to_label)
        print(f"  {name} 预测为: '{predicted_class}'")
