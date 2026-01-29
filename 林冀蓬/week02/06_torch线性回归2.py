import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

# 1. 生成sin函数的模拟数据
# 生成0到10之间的随机x值
X_numpy = np.random.rand(200, 1) * 10
# 计算sin(x)并添加一些噪声
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(200, 1)
# 转换为torch张量
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)

# 2. 定义多层神经网络模型
class MultiLayerNN(nn.Module):
    def __init__(self):
        super(MultiLayerNN, self).__init__()
        # 输入层到隐藏层1：1个输入特征 -> 64个神经元
        self.fc1 = nn.Linear(1, 64)
        # 隐藏层1到隐藏层2：64个神经元 -> 64个神经元
        self.fc2 = nn.Linear(64, 64)
        # 隐藏层2到隐藏层3：64个神经元 -> 32个神经元
        self.fc3 = nn.Linear(64, 32)
        # 隐藏层3到输出层：32个神经元 -> 1个输出
        self.fc4 = nn.Linear(32, 1)
        # 激活函数
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 前向传播
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 创建模型实例
model = MultiLayerNN()
print(model)
print("---" * 10)

# 3. 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

# 4. 训练模型
num_epochs = 5000
for epoch in range(num_epochs):
    # 前向传播：计算预测值
    y_pred = model(X)
    
    # 计算损失
    loss = criterion(y_pred, y)
    
    # 反向传播和优化
    optimizer.zero_grad()  # 清零梯度
    loss.backward()        # 反向传播计算梯度
    optimizer.step()       # 更新参数
    
    # 每100个epoch打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

# 5. 打印最终损失
print("\n训练完成！")
print(f"最终损失: {loss.item():.6f}")
print("---" * 10)

# 6. 可视化结果
# 生成用于绘制的均匀分布的x值
x_plot = torch.linspace(0, 10, 100).unsqueeze(1)
# 使用模型进行预测
with torch.no_grad():
    y_plot = model(x_plot)

plt.figure(figsize=(12, 8))
# 绘制原始数据点
plt.scatter(X_numpy, y_numpy, label='Raw data with noise', color='blue', alpha=0.6)
# 绘制真实的sin函数
plt.plot(x_plot.numpy(), np.sin(x_plot.numpy()), label='True sin(x)', color='green', linewidth=2)
# 绘制模型预测结果
plt.plot(x_plot.numpy(), y_plot.numpy(), label='NN Prediction', color='red', linewidth=3)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Multi-layer Neural Network Fitting sin(x)')
plt.legend()
plt.grid(True)
plt.savefig('sin_function_fitting.png')
plt.show()