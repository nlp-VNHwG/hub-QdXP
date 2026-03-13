import torch
import numpy as np
import matplotlib.pyplot as plt

# 生成0到2π的均匀分布数据（覆盖sin函数一个完整周期），添加少量噪声
X_numpy = np.linspace(0, 2 * np.pi, 200).reshape(-1, 1)  # (200, 1)，更多样本提升拟合效果
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(*X_numpy.shape)  # sin曲线 + 高斯噪声

# 转换为torch张量（保持原逻辑）
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("sin函数数据生成完成。")
print("---" * 10)

class SinFittingNet(torch.nn.Module):
    def __init__(self, input_dim=1, hidden_dims=[32, 16], output_dim=1):
        """
        多层全连接网络，用于拟合sin非线性函数
        :param input_dim: 输入维度（X是1维）
        :param hidden_dims: 隐藏层维度列表，多层提取非线性特征
        :param output_dim: 输出维度（y是1维）
        """
        super(SinFittingNet, self).__init__()
        self.layers = torch.nn.ModuleList()
        # 输入层 → 第一层隐藏层
        self.layers.append(torch.nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(torch.nn.ReLU())  # 激活函数引入非线性
        # 中间隐藏层（支持多层）
        for i in range(1, len(hidden_dims)):
            self.layers.append(torch.nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            self.layers.append(torch.nn.ReLU())
        # 最后一层隐藏层 → 输出层
        self.layers.append(torch.nn.Linear(hidden_dims[-1], output_dim))

    def forward(self, x):
        """前向传播：逐层计算"""
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

# 初始化模型（2层隐藏层，32→16节点，适配sin非线性拟合）
model = SinFittingNet(input_dim=1, hidden_dims=[32, 16], output_dim=1)
print("多层神经网络初始化完成：")
print(model)
print("---" * 10)

loss_fn = torch.nn.MSELoss()  # 回归任务仍用MSE损失
# 优化器传入模型所有参数（替代原手动定义的a、b）
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 改用Adam优化器，收敛更快

num_epochs = 2000  # 增加epoch，确保非线性拟合收敛
loss_history = []  # 记录loss变化，用于可视化
for epoch in range(num_epochs):
    # 前向传播：通过模型计算预测值（替代原手动a*X+b）
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)
    loss_history.append(loss.item())

    # 反向传播和优化（原逻辑不变）
    optimizer.zero_grad()  # 清空梯度
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新网络参数

    # 每100个epoch打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

print("\n训练完成！")
with torch.no_grad():  # 禁用梯度计算，提升效率
    y_predicted = model(X).numpy()  # 转换为numpy数组用于绘图

plt.figure(figsize=(15, 6))

# 子图1：训练loss变化
plt.subplot(1, 2, 1)
plt.plot(loss_history, label='Training Loss', color='green')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.grid(True)

# 子图2：sin函数拟合效果
plt.subplot(1, 2, 2)
# 绘制原始sin曲线（无噪声）
plt.plot(X_numpy, np.sin(X_numpy), label='Pure sin(x)', color='blue', linewidth=2, linestyle='--')
# 绘制带噪声的原始数据
plt.scatter(X_numpy, y_numpy, label='Raw data (sin + noise)', color='orange', alpha=0.5, s=10)
# 绘制模型拟合曲线
plt.plot(X_numpy, y_predicted, label='Model Fitting', color='red', linewidth=2)
plt.xlabel('X (0 ~ 2π)')
plt.ylabel('y = sin(x)')
plt.title('Sin Function Fitting with Multi-Layer Network')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
