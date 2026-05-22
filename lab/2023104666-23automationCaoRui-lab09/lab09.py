import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import random_split, DataLoader

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix
import seaborn as sns


results_dir = "./lab09/results"
os.makedirs(results_dir, exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("使用设备:", device)

# =========================
# 数据预处理
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


full_train_dataset = torchvision.datasets.MNIST(
    root='./lab09/data',
    train=True,
    download=False,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root='./lab09/data',
    train=False,
    download=False,
    transform=transform
)

# =========================
# 划分训练验证集
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
# CNN模型
# =========================
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)

        self.conv = nn.Sequential(

            self.conv1,

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),

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

# =========================
# 训练函数
# =========================
def train_model(optimizer_name='Adam', lr=0.001):

    model = CNN().to(device)

    criterion = nn.CrossEntropyLoss()

    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)

    elif optimizer_name == 'Momentum':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = 5

    train_losses = []
    val_losses = []

    train_accs = []
    val_accs = []

    for epoch in range(epochs):

        # =====================
        # 训练
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

        train_accs.append(train_acc)

        # =====================
        # 验证
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

        val_accs.append(val_acc)

        print(f"{optimizer_name} Epoch {epoch+1}")

    # =====================
    # 测试
    # =====================
    model.eval()

    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    wrong_images = []
    wrong_true = []
    wrong_pred = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 错误分类
            wrong_mask = predicted != labels

            for i in range(len(images)):

                if wrong_mask[i] and len(wrong_images) < 8:

                    wrong_images.append(images[i].cpu())
                    wrong_true.append(labels[i].item())
                    wrong_pred.append(predicted[i].item())

    test_acc = 100 * correct / total

    return model, train_losses, val_losses, train_accs, val_accs, test_acc, all_labels, all_preds, wrong_images, wrong_true, wrong_pred

# =========================
# 优化器对比
# =========================
optimizers = ['SGD', 'Momentum', 'Adam']

optimizer_results = {}

for opt in optimizers:

    model, train_losses, val_losses, train_accs, val_accs, test_acc, all_labels, all_preds, wrong_images, wrong_true, wrong_pred = train_model(opt)

    optimizer_results[opt] = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'test_acc': test_acc
    }

    print(f"{opt} Test Accuracy: {test_acc:.2f}%")

# =========================
# 绘制优化器Loss曲线
# =========================
plt.figure(figsize=(8,5))

for opt in optimizers:

    plt.plot(optimizer_results[opt]['val_losses'], label=opt)

plt.title("Optimizer Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig(f"{results_dir}/optimizer_loss.png")
plt.close()

# =========================
# 绘制优化器Accuracy曲线
# =========================
plt.figure(figsize=(8,5))

for opt in optimizers:

    plt.plot(optimizer_results[opt]['val_accs'], label=opt)

plt.title("Optimizer Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()

plt.savefig(f"{results_dir}/optimizer_accuracy.png")
plt.close()

# =========================
# 学习率对比
# =========================
learning_rates = [0.1, 0.01, 0.001]

lr_results = {}

for lr in learning_rates:

    model, train_losses, val_losses, train_accs, val_accs, test_acc, _, _, _, _, _ = train_model('Adam', lr)

    lr_results[str(lr)] = {
        'val_losses': val_losses,
        'val_accs': val_accs
    }

# =========================
# 学习率Loss曲线
# =========================
plt.figure(figsize=(8,5))

for lr in learning_rates:

    plt.plot(lr_results[str(lr)]['val_losses'], label=f"lr={lr}")

plt.title("Learning Rate Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.savefig(f"{results_dir}/learning_rate_loss.png")
plt.close()

# =========================
# 学习率Accuracy曲线
# =========================
plt.figure(figsize=(8,5))

for lr in learning_rates:

    plt.plot(lr_results[str(lr)]['val_accs'], label=f"lr={lr}")

plt.title("Learning Rate Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()

plt.savefig(f"{results_dir}/learning_rate_accuracy.png")
plt.close()

# =========================
# 卷积核可视化
# =========================
weights = model.conv1.weight.data.cpu()

fig, axes = plt.subplots(2,4, figsize=(8,4))

for i, ax in enumerate(axes.flat):

    kernel = weights[i,0,:,:]

    ax.imshow(kernel, cmap='gray')

    ax.axis('off')

plt.tight_layout()

plt.savefig(f"{results_dir}/conv_kernels.png")
plt.close()

# =========================
# Feature Map可视化
# =========================
images, labels = next(iter(test_loader))

image = images[0].unsqueeze(0).to(device)

with torch.no_grad():

    feature_maps = model.conv1(image)

feature_maps = feature_maps.cpu()

fig, axes = plt.subplots(2,4, figsize=(8,4))

for i, ax in enumerate(axes.flat):

    fmap = feature_maps[0,i,:,:]

    ax.imshow(fmap, cmap='gray')

    ax.axis('off')

plt.tight_layout()

plt.savefig(f"{results_dir}/feature_maps.png")
plt.close()

# =========================
# 错误分类样本
# =========================
fig, axes = plt.subplots(2,4, figsize=(10,5))

for i, ax in enumerate(axes.flat):

    img = wrong_images[i].squeeze().numpy()

    ax.imshow(img, cmap='gray')

    ax.set_title(f"T:{wrong_true[i]} P:{wrong_pred[i]}")

    ax.axis('off')

plt.tight_layout()

plt.savefig(f"{results_dir}/wrong_predictions.png")
plt.close()

# =========================
# 混淆矩阵
# =========================
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8,6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title("Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("True")

plt.savefig(f"{results_dir}/confusion_matrix.png")
plt.close()

print("\n实验完成！")