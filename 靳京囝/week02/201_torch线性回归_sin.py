import torch
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt

# 1. 生成模拟数据
# 生成x：范围在0到2π之间，1000个点（更多点让曲线更平滑）
X_numpy = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)  # 形状 (1000, 1)
# 生成y：sin(x) + 少量高斯噪声（模拟真实数据）
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(*X_numpy.shape)

X = torch.from_numpy(X_numpy).float() # torch 中 所有的计算 通过tensor 计算
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)

# 2. 定义非线性模型（多层感知机，拟合正弦曲线）
# 线性模型无法拟合sin函数，需用隐藏层引入非线性
class SinModel(torch.nn.Module):
    def __init__(self):
        super(SinModel, self).__init__()
        # 输入层(1维) → 隐藏层(8节点) → 隐藏层(16节点) → 输出层(1维)
        self.fc1 = torch.nn.Linear(1, 8)
        self.relu = torch.nn.ReLU()  # 激活函数引入非线性
        self.fc2 = torch.nn.Linear(8, 16)
        self.fc3 = torch.nn.Linear(16, 1)

    def forward(self, x):
        # 前向传播：数据流过各层
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# 初始化模型
model = SinModel()
print("模型结构：")
print(model)
print("---" * 10)

# 3. 定义损失函数和优化器
# 损失函数仍然是均方误差 (MSE)。
loss_fn = torch.nn.MSELoss() # 回归任务
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam优化器

# 4. 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播：手动计算
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度， torch 梯度 累加
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 关闭梯度计算，提升速度
with torch.no_grad():
    y_predicted = model(X).numpy()  # 转换为numpy数组用于绘图

# 6. 绘制结果
plt.figure(figsize=(10, 6))
# plt.subplot(1, 1, 1)
plt.scatter(X_numpy, y_numpy, label='Raw data sin(x)', color='blue', alpha=0.6, s=5)
plt.plot(X_numpy, y_predicted, label='Fitted sin curve', color='red', linewidth=2)
plt.plot(X_numpy, np.sin(X_numpy), label='Pure sin(x)', color='green', linestyle='--', linewidth=2)
plt.xlabel('X (0 ~ 2π)')
plt.ylabel('y')
plt.title('Sin Function Fitting Result')
plt.legend()
plt.grid(True)
plt.show()
