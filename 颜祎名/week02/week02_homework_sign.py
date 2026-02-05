import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成 Sin 函数模拟数据
X_numpy = np.linspace(-5, 5, 500).reshape(-1, 1)  # 生成 -5 到 5 之间的 500 个点
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(500, 1)  # sin(x) 加一点噪声
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()


# 2. 定义多层神经网络模型
class SinNet(nn.Module):
    def __init__(self):
        super(SinNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),  # 输入 1 维 (x)，输出 64 维
            nn.ReLU(),  # 必须有激活函数，否则无法拟合曲线
            nn.Linear(64, 64),  # 隐藏层
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出 1 维 (y)
        )

    def forward(self, x):
        return self.net(x)


model = SinNet()
# 使用 Adam 优化器，拟合效果更好
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()  # 这里的损失函数使用均方误差

# 3. 训练过程
epochs = 1000
for epoch in range(epochs):
    y_pred = model(X)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 4. 可视化结果
model.eval()
with torch.no_grad():
    y_final = model(X)

plt.figure(figsize=(10, 5))
plt.scatter(X_numpy, y_numpy, s=10, label='Target', alpha=0.5)
plt.plot(X_numpy, y_final.numpy(), color='red', label='Learned Sin', linewidth=3)
plt.legend()
plt.title("Fitting Sine Wave with Multi-layer Neural Network")
plt.show()