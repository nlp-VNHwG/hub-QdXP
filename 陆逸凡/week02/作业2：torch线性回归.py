import torch
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt
import math

'''
输入的函数是sin，然后微调模型
'''

# 生成从 -10 到 10（包含两端）的 1000 个等间隔数字
# 将一维数组转换为二维数组，形状为 (1000, 1)
X_numpy = np.linspace(-10, 10, 2000).reshape(-1, 1)
# 通过sin函数生成对应的y值，并添加一些随机噪声
y_numpy = np.sin(X_numpy) +  0.1 * np.random.randn(2000, 1)
X = torch.from_numpy(X_numpy).float() # torch 中 所有的计算 通过tensor 计算
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)
# 2. 定义模型参数
class SinModel(torch.nn.Module):    
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(SinModel, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_dim)
        )
    def forward(self,x):
        output=self.network(x)
        return output

# 3. 生成模型对象，定义损失函数和优化器
model = SinModel(input_dim=1,hidden_dim=64,output_dim=1)
# 损失函数仍然是均方误差 (MSE)。
loss_fn = torch.nn.MSELoss() # 回归任务
# 优化器现在直接传入我们手动创建的参数 [a, b]。
# PyTorch 会自动根据这些参数的梯度来更新它们。
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # 优化器，基于 a b 梯度 自动更新

# 4. 训练模型
num_epochs = 1000
losses = []  # 用于记录每个 epoch 的损失值
for epoch in range(num_epochs):
    
    y_pred = model(X)
    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度， torch 梯度 累加
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数
    losses.append(loss.item())
    # 每100个 epoch 打印一次损失
    if epoch % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 打印最终学到的参数
print("\n训练完成！")


# 6. 绘制结果
# 使用最终学到的参数进行预测
with torch.no_grad():
    y_predicted = model(X)

plt.figure(figsize=(16, 6))
#打印loss 曲线  
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('turn')
plt.ylabel('loss')
plt.title('loss curve')
plt.grid(True)
#打印拟合曲线
plt.subplot(1, 2, 2)
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.plot(X_numpy, y_predicted, label=f'Model: y', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Sine Function Fitting using Neural Network')
plt.legend()
plt.grid(True)
plt.show()
