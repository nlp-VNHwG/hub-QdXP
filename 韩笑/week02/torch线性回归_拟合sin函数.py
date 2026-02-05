import torch
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt
import torch.nn as nn

# 1. 生成模拟数据
#含有100个数据点的数组，作为正弦函数的输入 从0-1 步长为0.01
X_numpy = np.arange(0.0,1.0,0.01)
# 0-1之间的x *2Π 从单位间隔转化为弧度值
#将x映射到正弦函数的一个完整的周期上
Y_numpy = np.sin(2*np.pi*X_numpy)
#通过reshape转化为100*1的数组，代表100个坐标 100个训练集
X_numpy=X_numpy.reshape(100,1)
Y_numpy=Y_numpy.reshape(100,1)
# 转化为tensor
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(Y_numpy).float()

print("数据生成完成。")
print("---" * 10)


#2.定义多层网络模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, output_size):
        super(MLP, self).__init__()

        # 使用 nn.Sequential 封装多层网络
        # 这是一种简洁且常用的方式，可以方便地组织和查看网络结构
        self.network = nn.Sequential(
            # 第1层：从 input_size 到 hidden_size1
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(), # 增加模型的复杂度，非线性
            # 输出层：从 hidden_size1 到 output_size
            nn.Linear(hidden_size1, output_size)
        )

    def forward(self, x):
        return self.network(x)

#模型参数
input_size = 1
hidden_size1 = 64
output_size = 1

# 实例化模型
model = MLP(input_size, hidden_size1, output_size)
print("模型结构:\n", model)


# 3. 定义损失函数和优化器
# 损失函数仍然是均方误差 (MSE)。
loss_fn = torch.nn.MSELoss() # 回归任务
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. 训练模型
num_epochs = 10000
for epoch in range(num_epochs):
    # 前向传播：使用模型预测
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度， torch 梯度 累加
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    # 每1000个 epoch 打印一次损失
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5.使用训练好的模型进行预测
with torch.no_grad():
    y_predicted = model(X).numpy()

# 6.绘制结果
plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, Y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.plot(X_numpy, y_predicted, label='Model', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
