import torch
import numpy as np  # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt
import math

'''
输入的函数是sin，然后微调模型
使用神经网络来拟合正弦函数sin(x)
'''

# 1. 生成训练数据
# np.linspace(start, stop, num): 生成从-10到10的等间隔数字
# reshape(-1, 1): 将一维数组转换为二维数组，形状为(样本数, 特征数)
# -10: 起始值, 10: 结束值, 2000: 样本数量
X_numpy = np.linspace(-10, 10, 2000).reshape(-1, 1)

# 生成目标值: sin(x) + 噪声
# np.sin(X_numpy): 计算正弦函数值
# np.random.randn(2000, 1): 生成形状为(2000,1)的正态分布随机噪声
# 0.1: 噪声的幅度，控制噪声大小
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(2000, 1)

# 将NumPy数组转换为PyTorch张量
# torch.from_numpy(): 将NumPy数组转换为PyTorch张量
# .float(): 转换为浮点类型，PyTorch默认使用float32进行计算
X = torch.from_numpy(X_numpy).float()  # torch中所有的计算通过tensor计算
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)  # 打印分隔线

# 2. 定义神经网络模型类
class SinModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        初始化神经网络模型
        
        参数:
        input_dim: 输入维度 (这里是1，因为x是标量)
        hidden_dim: 第一个隐藏层的神经元数量
        output_dim: 输出维度 (这里是1，因为预测y是标量)
        """
        # 调用父类的初始化方法
        super(SinModel, self).__init__()
        
        # 使用nn.Sequential定义神经网络层
        # Sequential是一个容器，按顺序执行其中的层
        self.network = torch.nn.Sequential(
            # 第一层: 输入层 -> 隐藏层
            # Linear(输入维度, 输出维度): 全连接层，进行线性变换
            torch.nn.Linear(input_dim, hidden_dim),
            
            # 激活函数: ReLU (Rectified Linear Unit)
            # 引入非线性，使神经网络能够拟合非线性函数
            torch.nn.ReLU(),
            
            # 第二层: 隐藏层 -> 隐藏层 (128个神经元)
            torch.nn.Linear(hidden_dim, 128),
            torch.nn.ReLU(),
            
            # 第三层: 隐藏层 -> 隐藏层 (128个神经元)
            # 保持相同维度，加深网络深度
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            
            # 第四层: 隐藏层 -> 隐藏层 (64个神经元)
            # 减少神经元数量，进行特征压缩
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            
            # 第五层: 隐藏层 -> 输出层
            # 输出层，将特征映射到最终输出
            torch.nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        """
        前向传播函数
        
        参数:
        x: 输入张量
        
        返回:
        output: 网络输出
        """
        # 将输入x传递过整个网络
        output = self.network(x)
        return output

# 3. 创建模型实例，定义损失函数和优化器
# 创建模型对象，指定输入维度=1，隐藏层维度=64，输出维度=1
model = SinModel(input_dim=1, hidden_dim=64, output_dim=1)

# 定义损失函数: 均方误差 (Mean Squared Error)
# MSE常用于回归任务，计算预测值与真实值之间的平方差均值
loss_fn = torch.nn.MSELoss()  # 回归任务

# 定义优化器: Adam优化器
# model.parameters(): 获取模型所有可训练参数
# lr=0.01: 学习率，控制参数更新的步长
# Adam优化器结合了动量和自适应学习率的优点
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. 训练模型
num_epochs = 1000  # 训练轮数
losses = []  # 用于记录每个epoch的损失值，用于绘制损失曲线

# 训练循环
for epoch in range(num_epochs):
    # 前向传播: 通过模型计算预测值
    y_pred = model(X)
    
    # 计算损失: 比较预测值和真实值
    loss = loss_fn(y_pred, y)
    
    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度，防止梯度累积
    loss.backward()        # 反向传播，计算梯度
    optimizer.step()       # 更新参数，根据梯度调整参数值
    
    # 记录当前epoch的损失值
    losses.append(loss.item())
    
    # 每100个epoch打印一次损失
    if epoch % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 训练完成
print("\n训练完成！")

# 6. 可视化结果
# 使用训练好的模型进行预测 (不计算梯度，节省内存)
with torch.no_grad():  # 禁用梯度计算
    y_predicted = model(X)

# 创建图形窗口，设置大小
plt.figure(figsize=(16, 6))

# 子图1: 损失曲线
plt.subplot(1, 2, 1)  # 1行2列的第1个子图
plt.plot(losses)  # 绘制损失值随训练轮数的变化
plt.xlabel('Epoch')  # x轴标签: 训练轮数
plt.ylabel('Loss')   # y轴标签: 损失值
plt.title('Training Loss Curve')  # 子图标题
plt.grid(True)  # 显示网格

# 子图2: 拟合结果
plt.subplot(1, 2, 2)  # 1行2列的第2个子图
# 绘制原始数据点 (散点图)
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
# 绘制模型预测曲线 (线图)
plt.plot(X_numpy, y_predicted, label=f'Model prediction', color='red', linewidth=2)
# 绘制真实的正弦函数曲线作为对比
plt.plot(X_numpy, np.sin(X_numpy), label='True sin(x)', color='green', linestyle='--', linewidth=1.5)

plt.xlabel('X')  # x轴标签
plt.ylabel('y')  # y轴标签
plt.title('Sine Function Fitting using Neural Network')  # 子图标题
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格

# 显示图形
plt.show()