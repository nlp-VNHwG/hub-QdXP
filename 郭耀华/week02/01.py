import torch
import numpy as np
import matplotlib.pyplot as plt
import math

X_numpy = np.random.rand(100, 1) * 10


# y_numpy = 2 * X_numpy + 1 + np.random.randn(100, 1)
y_numpy = 2*np.sin(X_numpy) + np.random.randn(100, 1)*0.1
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)

a = torch.randn(1, requires_grad=True, dtype=torch.float)
b = torch.randn(1, requires_grad=True, dtype=torch.float)

print(f"初始参数 a: {a.item():.4f}")
print(f"初始参数 b: {b.item():.4f}")
print("---" * 10)


loss_fn = torch.nn.MSELoss() # 回归任务

optimizer = torch.optim.SGD([a, b], lr=0.0005) # 优化器，基于 a b 梯度 自动更新

num_epochs = 1000
for epoch in range(num_epochs):
    y_pred = a * np.sin(X) + b

    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("\n训练完成！")
a_learned = a.item()
b_learned = b.item()
print(f"拟合的斜率 a: {a_learned:.4f}")
print(f"拟合的截距 b: {b_learned:.4f}")
print("---" * 10)

with torch.no_grad():
    y_predicted = a_learned * X + b_learned

plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.plot(X_numpy, y_predicted, label=f'Model: y = {a_learned:.2f}x + {b_learned:.2f}', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
