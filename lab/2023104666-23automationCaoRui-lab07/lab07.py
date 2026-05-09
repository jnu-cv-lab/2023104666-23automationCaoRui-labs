
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import os

# 创建输出图片的目录
os.makedirs("results", exist_ok=True)

# ========== 任务1：数据准备 ==========
print("=" * 50)
print("任务1：数据准备")
digits = load_digits()
X = digits.images          # 形状: (1797, 8, 8)
y = digits.target          # 标签 0~9

n_samples, h, w = X.shape
print(f"图像数量: {n_samples}")
print(f"图像大小: {h}×{w}")
print(f"类别标签: {np.unique(y)}")

# 显示若干张样本图像
fig, axes = plt.subplots(2, 5, figsize=(8, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(X[i], cmap='gray')
    ax.set_title(f"Label: {y[i]}")
    ax.axis('off')
plt.suptitle("Sample Images from Digits Dataset")
plt.tight_layout()
plt.savefig("results/样本图像示例.png", dpi=150)
plt.show()

# ========== 任务2：数据划分 ==========
print("\n" + "=" * 50)
print("任务2：数据划分")
X_flat = digits.data  # shape (1797, 64)
X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y, test_size=0.25, random_state=42, stratify=y
)
print(f"训练集样本数: {X_train.shape[0]}")
print(f"测试集样本数: {X_test.shape[0]}")
print("说明：训练集用于学习模型参数，测试集用于评估模型泛化能力。")

# ========== 任务3：特征表示 ==========
print("\n" + "=" * 50)
print("任务3：特征表示")
print("每张8×8图像按行展开成64维向量（像素灰度值）。")
print("传统机器学习方法通常要求输入为固定长度的特征向量，无法直接处理二维图像结构。")
print("原始像素特征的优点：简单、直接，保留全部视觉信息；")
print("局限：缺乏平移/旋转/尺度不变性，对噪声敏感，维度可能较高。")

# ========== 任务4：模型训练 ==========
print("\n" + "=" * 50)
print("任务4：模型训练与评估")

models = {
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
    "SVM": SVC(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name}: 测试准确率 = {acc:.4f}")

# ========== 任务5：结果比较 ==========
print("\n" + "=" * 50)
print("任务5：准确率比较表格")
df_result = pd.DataFrame(list(results.items()), columns=["模型", "测试准确率"])
df_result["测试准确率"] = df_result["测试准确率"].apply(lambda x: f"{x:.4f}")
print(df_result.to_string(index=False))

best_model_name = max(results, key=results.get)
worst_model_name = min(results, key=results.get)
print(f"\n准确率最高: {best_model_name} ({results[best_model_name]:.4f})")
print(f"准确率最低: {worst_model_name} ({results[worst_model_name]:.4f})")
print("不同模型间表现差异较为明显（如决策树与SVM之间），原因在于模型假设与复杂度不同。")
print("线性模型（如逻辑回归）对非线性特征拟合能力有限；树模型易过拟合；SVM和随机森林泛化能力较好。")

# ========== 任务6：错误样本分析 ==========
print("\n" + "=" * 50)
print("任务6：错误样本分析（选择 Random Forest）")

best_model = models["Random Forest"]
y_pred_best = best_model.predict(X_test)

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=digits.target_names)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title("Confusion Matrix - Random Forest")
plt.savefig("results/混淆矩阵_随机森林.png", dpi=150)
plt.show()

# 找出错误分类样本
errors = np.where(y_pred_best != y_test)[0]
print(f"错误样本数量: {len(errors)} / {len(y_test)}")

# 分析哪些数字最易混淆
error_pairs = {}
for idx in errors:
    true_label = y_test[idx]
    pred_label = y_pred_best[idx]
    if true_label != pred_label:
        pair = (true_label, pred_label)
        error_pairs[pair] = error_pairs.get(pair, 0) + 1

# 按错误次数排序，取前5组
sorted_pairs = sorted(error_pairs.items(), key=lambda x: x[1], reverse=True)
print("\n最常混淆的数字组合 (真实 → 预测) 及次数:")
for (true, pred), count in sorted_pairs[:5]:
    print(f"  {true} → {pred}: {count} 次")

# 展示若干错误分类样本
n_show = min(6, len(errors))
fig, axes = plt.subplots(1, n_show, figsize=(12, 3))
for i, ax in enumerate(axes.flat):
    idx = errors[i]
    ax.imshow(X_test[idx].reshape(8, 8), cmap='gray')
    ax.set_title(f"True:{y_test[idx]}\nPred:{y_pred_best[idx]}")
    ax.axis('off')
plt.suptitle("Misclassified Samples (Random Forest)")
plt.tight_layout()
plt.savefig("results/错误分类样本_随机森林.png", dpi=150)
plt.show()

print("\n分析：错误样本通常笔画细、形状不规范或与其它数字相似，比如 8 与 3、5 与 3 等。")
print("像素级别特征缺乏形状不变性，模型容易受局部噪声或书写风格影响。")

print("\n" + "=" * 50)
print("实验完成，所有结果图片已保存至results文件夹。")