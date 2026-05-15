import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
import numpy as np

# =========================
# 创建 results 文件夹
# =========================
results_dir = "lab08/results"
os.makedirs(results_dir, exist_ok=True)

# =========================
# 判断设备
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("使用设备:", device)

# =========================
# 数据预处理
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# =========================
# 下载 MNIST 数据集
# =========================
full_train_dataset = torchvision.datasets.MNIST(
    root='./lab08/data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root='./lab08/data',
    train=False,
    download=True,
    transform=transform
)

# =========================
# 划分训练集和验证集
# =========================
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size, val_size]
)

# =========================
# DataLoader
# =========================
batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# =========================
# 显示样本图像
# =========================
classes = [str(i) for i in range(10)]

images, labels = next(iter(train_loader))

fig, axes = plt.subplots(2, 4, figsize=(10, 5))

for i, ax in enumerate(axes.flat):
    img = images[i].squeeze().numpy()

    ax.imshow(img, cmap='gray')
    ax.set_title(f"Label: {labels[i].item()}")
    ax.axis('off')

plt.tight_layout()

sample_path = os.path.join(results_dir, "dataset_samples.png")
plt.savefig(sample_path)
plt.close()

print("样本图像已保存")

# =========================
# 定义 CNN 网络
# =========================
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),

            nn.Linear(128, 10)
        )

    def forward(self, x):

        x = self.conv(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x

model = CNN().to(device)

print(model)

# =========================
# 损失函数和优化器
# =========================
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# =========================
# 训练参数
# =========================
epochs = 5

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

# =========================
# 开始训练
# =========================
for epoch in range(epochs):

    # =====================
    # 训练模式
    # =====================
    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)

    train_acc = 100 * correct / total

    train_losses.append(train_loss)

    train_accuracies.append(train_acc)

    # =====================
    # 验证模式
    # =====================
    model.eval()

    val_running_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (predicted == labels).sum().item()

    val_loss = val_running_loss / len(val_loader)

    val_acc = 100 * val_correct / val_total

    val_losses.append(val_loss)

    val_accuracies.append(val_acc)

    print(f"\nEpoch [{epoch+1}/{epochs}]")

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.2f}%")

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.2f}%")

# =========================
# 测试模型
# =========================
model.eval()

test_loss = 0
test_correct = 0
test_total = 0

all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        test_total += labels.size(0)

        test_correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_accuracy = 100 * test_correct / test_total

test_loss = test_loss / len(test_loader)

print("\n===== 测试结果 =====")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")

# =========================
# 绘制 Loss 曲线
# =========================
plt.figure(figsize=(8, 5))

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()

loss_path = os.path.join(results_dir, "loss_curve.png")

plt.savefig(loss_path)
plt.close()

print("Loss曲线已保存")

# =========================
# 绘制 Accuracy 曲线
# =========================
plt.figure(figsize=(8, 5))

plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Curve")
plt.legend()

acc_path = os.path.join(results_dir, "accuracy_curve.png")

plt.savefig(acc_path)
plt.close()

print("Accuracy曲线已保存")

# =========================
# 显示测试预测结果
# =========================
images, labels = next(iter(test_loader))

images = images.to(device)

outputs = model(images)

_, predicted = torch.max(outputs, 1)

images = images.cpu()

fig, axes = plt.subplots(2, 4, figsize=(10, 5))

for i, ax in enumerate(axes.flat):

    img = images[i].squeeze().numpy()

    ax.imshow(img, cmap='gray')

    true_label = labels[i].item()

    pred_label = predicted[i].item()

    ax.set_title(f"T:{true_label} P:{pred_label}")

    ax.axis('off')

plt.tight_layout()

pred_path = os.path.join(results_dir, "test_predictions.png")

plt.savefig(pred_path)

plt.close()

print("测试预测图已保存")

print("\n实验完成！")