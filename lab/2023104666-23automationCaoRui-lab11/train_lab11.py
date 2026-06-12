import os
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import seaborn as sns

# =========================
# 路径
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(
    BASE_DIR,
    "skeleton_data"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# 超参数
# =========================

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3

INPUT_DIM = 132
SEQ_LEN = 30

D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
FF_DIM = 256

NUM_CLASSES = 6

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("使用设备:", device)

# =========================
# 读取数据
# =========================

X_train = np.load(
    os.path.join(DATA_DIR, "X_train.npy")
)

y_train = np.load(
    os.path.join(DATA_DIR, "y_train.npy")
)

X_test = np.load(
    os.path.join(DATA_DIR, "X_test.npy")
)

y_test = np.load(
    os.path.join(DATA_DIR, "y_test.npy")
)

with open(
    os.path.join(DATA_DIR, "label_map.json"),
    "r",
    encoding="utf-8"
) as f:

    label_map = json.load(f)

id2label = {
    v:k for k,v in label_map.items()
}

# =========================
# Dataset
# =========================

class SkeletonDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]

train_loader = DataLoader(
    SkeletonDataset(X_train,y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    SkeletonDataset(X_test,y_test),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# =========================
# Transformer
# =========================

class SkeletonTransformer(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Linear(
            INPUT_DIM,
            D_MODEL
        )

        self.pos_embed = nn.Parameter(
            torch.randn(
                1,
                SEQ_LEN,
                D_MODEL
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=FF_DIM,
            dropout=0.1,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=NUM_LAYERS
        )

        self.classifier = nn.Sequential(
            nn.Linear(D_MODEL,64),
            nn.ReLU(),
            nn.Linear(64,NUM_CLASSES)
        )

    def forward(self,x):

        x = self.embedding(x)

        x = x + self.pos_embed

        x = self.encoder(x)

        x = x.mean(dim=1)

        x = self.classifier(x)

        return x

model = SkeletonTransformer().to(device)

print(model)

# =========================
# Loss & Optimizer
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

# =========================
# 训练
# =========================

train_losses = []
train_accs = []

best_acc = 0

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    all_pred = []
    all_true = []

    for X,y in train_loader:

        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        out = model(X)

        loss = criterion(out,y)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        pred = out.argmax(1)

        all_pred.extend(
            pred.cpu().numpy()
        )

        all_true.extend(
            y.cpu().numpy()
        )

    epoch_loss = running_loss/len(train_loader)

    epoch_acc = accuracy_score(
        all_true,
        all_pred
    )

    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)

    print(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"Loss={epoch_loss:.4f} "
        f"Acc={epoch_acc:.4f}"
    )

    # 测试

    model.eval()

    preds = []
    trues = []

    with torch.no_grad():

        for X,y in test_loader:

            X = X.to(device)

            out = model(X)

            pred = out.argmax(1)

            preds.extend(
                pred.cpu().numpy()
            )

            trues.extend(
                y.numpy()
            )

    test_acc = accuracy_score(
        trues,
        preds
    )

    print(
        f"Test Acc={test_acc:.4f}"
    )

    if test_acc > best_acc:

        best_acc = test_acc

        torch.save(
            model.state_dict(),
            os.path.join(
                MODEL_DIR,
                "best_model.pth"
            )
        )

# =========================
# Loss曲线
# =========================

plt.figure()

plt.plot(train_losses)

plt.title("Train Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "train_loss.png"
    )
)

plt.close()

# =========================
# Accuracy曲线
# =========================

plt.figure()

plt.plot(train_accs)

plt.title("Train Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "train_accuracy.png"
    )
)

plt.close()

# =========================
# 最终测试
# =========================

model.load_state_dict(
    torch.load(
        os.path.join(
            MODEL_DIR,
            "best_model.pth"
        ),
        map_location=device
    )
)

model.eval()

preds = []
trues = []

with torch.no_grad():

    for X,y in test_loader:

        X = X.to(device)

        out = model(X)

        pred = out.argmax(1)

        preds.extend(
            pred.cpu().numpy()
        )

        trues.extend(
            y.numpy()
        )

final_acc = accuracy_score(
    trues,
    preds
)

print("\n最终测试准确率:")
print(final_acc)

# =========================
# 混淆矩阵
# =========================

cm = confusion_matrix(
    trues,
    preds
)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "confusion_matrix.png"
    )
)

plt.close()

# =========================
# 分类报告
# =========================

report = classification_report(
    trues,
    preds,
    target_names=[
        id2label[i]
        for i in range(NUM_CLASSES)
    ]
)

with open(
    os.path.join(
        RESULT_DIR,
        "classification_report.txt"
    ),
    "w"
) as f:

    f.write(report)

print(report)

# =========================
# 单样本推理
# =========================

sample = torch.tensor(
    X_test[0:1],
    dtype=torch.float32
).to(device)

with torch.no_grad():

    out = model(sample)

    prob = torch.softmax(
        out,
        dim=1
    )

    conf, pred = torch.max(
        prob,
        dim=1
    )

pred_class = id2label[
    pred.item()
]

with open(
    os.path.join(
        RESULT_DIR,
        "inference_result.txt"
    ),
    "w"
) as f:

    f.write(
        f"Predicted class: {pred_class}\n"
    )

    f.write(
        f"Confidence: {conf.item():.4f}\n"
    )

print("\n推理结果:")
print(pred_class)
print(conf.item())

print("\n实验完成")