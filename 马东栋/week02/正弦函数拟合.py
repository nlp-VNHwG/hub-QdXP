import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

# 1. 生成模拟数据 (与之前相同)
X_numpy = np.random.rand(100, 1) * 10
y_numpy = np.sin(X_numpy) + np.random.randn(100, 1)
# y_numpy = 2 * X_numpy + 1 + np.random.randn(100, 1)
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)


# 3. 定义学习率
learning_rate = 0.01

# 定义正弦函数模型
class SinusoidalModel(nn.Module):
    def __init__(self):
        super(SinusoidalModel, self).__init__()
        self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.w = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.f = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))


    def forward(self, x):
        return self.a * torch.sin(self.w * x + self.f)

model = SinusoidalModel()

# 4. 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播：计算预测值 y_pred
    y_pred = model(X)

    # 手动计算 MSE 损失
    loss = torch.mean((y_pred - y)**2)

    # 手动反向传播：计算 a 和 b 的梯度
    # PyTorch 的自动求导会帮我们计算，我们只需要调用 loss.backward()


    loss.backward()

    # 手动更新参数
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad


    # 但在这里，我们手动计算梯度，因此需要确保梯度清零
    model.zero_grad()
    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 打印最终学到的参数
print("\n训练完成！")
a_learned = model.a.item()
w_learned = model.w.item()
f_learned = model.f.item()

print(f"拟合的振幅 a: {a_learned:.4f}")
print(f"拟合的频率 w: {f_learned:.4f}")
print(f"拟合的相位 f: {f_learned:.4f}")
print("---" * 10)

# 6. 绘制结果
with torch.no_grad():
    y_predicted = model(X)

sorted_indices = np.argsort(X_numpy.flatten())
X_sorted = X_numpy[sorted_indices]
y_pred_sorted = y_predicted.numpy()[sorted_indices]

plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.plot(X_sorted, y_pred_sorted, 'r-', label='Fitted curve', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()