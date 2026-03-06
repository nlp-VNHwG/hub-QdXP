import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

dataset = pd.read_csv("../Week01/作业2/dataset.csv", sep="\t", header=None)
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

index_to_label = {i: label for label, i in label_to_index.items()}
max_len = 40

class CharLSTMDataset(Dataset):
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

dataset = CharLSTMDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        out, hidden = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        out, (hidden, cell) = self.lstm(embedded)
        return self.fc(hidden.squeeze(0))


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        out, hidden = self.gru(embedded)
        return self.fc(hidden.squeeze(0))



def train_model(model, dataloader, epochs=4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss = {total_loss/len(dataloader):.4f}")

    return model


def classify_text(text, model, char_to_index, max_len, index_to_label):
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    pred_idx = torch.argmax(output, dim=1).item()
    return index_to_label[pred_idx]


embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)

models = {
    "RNN": RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim),
    "LSTM": LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim),
    "GRU": GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim),
}

results = {}

for name, model in models.items():
    print(f"\n==================== 训练 {name} ====================")
    trained_model = train_model(model, dataloader)

    # 计算训练集精度
    all_preds, all_labels = [], []
    for inputs, labels in dataloader:
        preds = trained_model(inputs).argmax(dim=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    results[name] = acc
    print(f"{name} 训练集精度: {acc:.4f}")


test_texts = [
    "帮我导航到北京",
    "查询明天北京的天气"
]

for name, model in models.items():
    print(f"\n--- {name} 模型预测 ---")
    for t in test_texts:
        print(t, "→", classify_text(t, model, char_to_index, max_len, index_to_label))

print("\n====== 精度对比 ======")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")
