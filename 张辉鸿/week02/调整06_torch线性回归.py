import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成 sin 函数数据（修改这里）
X_numpy = np.random.rand(100, 1) * 2 * np.pi  # [0, 2π] 范围
y_numpy = np.sin(X_numpy)  # sin 函数（改动点）

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)

# 2. 定义多层网络（sin是非线性的，需要多层网络）
model = torch.nn.Sequential(
    torch.nn.Linear(1, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 1)
)

print("模型结构：", model)
print("---" * 10)

# 3. 定义损失函数和优化器
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. 训练模型
num_epochs = 2000
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

# 5. 打印结果
print("\n训练完成！")
print(f"最终损失: {loss.item():.6f}")
print("---" * 10)

# 6. 绘制结果
with torch.no_grad():
    # 用于绘制平滑曲线
    X_plot = torch.linspace(0, 2 * np.pi, 200).reshape(-1, 1)
    y_predicted = model(X_plot)

plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data (sin)', color='blue', alpha=0.6)
plt.plot(X_plot.numpy(), y_predicted.numpy(), label='Fitted curve', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
