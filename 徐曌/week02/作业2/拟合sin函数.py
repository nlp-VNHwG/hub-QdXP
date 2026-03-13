import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 1. 生成 sin(x) 数据
def generate_sin_data(n_samples=1000):
    X = np.linspace(-2 * np.pi, 2 * np.pi, n_samples).reshape(-1, 1)
    y = np.sin(X)
    return torch.from_numpy(X).float(), torch.from_numpy(y).float()

# 2. 构建多层神经网络
class SinApproximator(nn.Module):
    def __init__(self):
        super(SinApproximator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

# 3. 训练函数
def train_model(model, X, y, num_epochs=200, lr=0.01, print_every=100):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []

    for epoch in range(num_epochs):
        model.train()
        y_pred = model(X)
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if (epoch + 1) % print_every == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")
    return loss_history

# 4. 可视化函数
def plot_results(X, y_true, y_pred, loss_history):
    plt.figure(figsize=(12, 5))

    # 左图：拟合曲线
    plt.subplot(1, 2, 1)
    plt.plot(X.numpy(), y_true.numpy(), label='True sin(x)', color='blue')
    plt.plot(X.numpy(), y_pred.detach().numpy(), label='Predicted', color='red')
    plt.title("sin(x) ")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    # 右图：loss 曲线
    plt.subplot(1, 2, 2)
    plt.plot(loss_history, label='Loss', color='green')
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# 5. 主流程
if __name__ == "__main__":
    X, y = generate_sin_data()
    model = SinApproximator()
    print("开始训练模型拟合 sin(x)...")
    loss_history = train_model(model, X, y, num_epochs=300, lr=0.01)
    print("训练完成！")
    with torch.no_grad():
        y_pred = model(X)
    plot_results(X, y, y_pred, loss_history)
