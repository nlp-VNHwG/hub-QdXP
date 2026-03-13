import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import torch.optim as optim
# 1. 生成模拟数据 (y=sinx)
x = torch.linspace(-10, 10, 400)
x = x.unsqueeze(1)
y = torch.sin(x)

import torch.nn as nn

class SinActivationModel(nn.Module):
    def __init__(self):
        super(SinActivationModel, self).__init__()
        self.fc1 = nn.Linear(1, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = SinActivationModel()

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 4000

for epoch in range(num_epochs):
    # --- 前向传播 ---
    y_pred = model(x)
    loss = criterion(y_pred, y)

    # --- 反向传播 ---
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("训练完成！")

model.eval()

prediction = model(x).detach().numpy()
plt.figure(figsize=(10, 6))
plt.scatter(x.numpy(), y.numpy(), label='True Data (Sin)', color='blue', alpha=0.5)
plt.plot(x.numpy(), prediction, label='Neural Network Fitting', color='red', linewidth=3)

plt.legend()
plt.title(f'Fitting Sin(x) - Final Loss: {loss.item():.4f}')
plt.show()
