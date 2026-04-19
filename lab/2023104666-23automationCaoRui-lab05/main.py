import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

# ===================== 修复中文标题乱码（只改这里）=====================
plt.rcParams['font.family'] = ['DejaVu Sans']  # 系统自带字体
plt.rcParams['axes.unicode_minus'] = False

# ===================== 1. 读取图片 =====================
test_img = cv2.imread("test_geo.jpg")
distort_img = cv2.imread("perspective_distort.jpg")

test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
distort_img_rgb = cv2.cvtColor(distort_img, cv2.COLOR_BGR2RGB)
h, w = test_img.shape[:2]

print("========== Lab05 图像几何变换实验 ==========")
print("1. 相似变换、仿射变换、透视变换对比")
print("2. 几何性质变化规律总结")
print("3. 透视畸变图像校正")

# ===================== 2. 三种几何变换实现 =====================
# ---------- 2.1 相似变换
scale = 0.8
angle = 30
tx, ty = 50, 30
similarity_M = cv2.getRotationMatrix2D((w//2, h//2), angle, scale)
similarity_M[:, 2] += [tx, ty]
img_similar = cv2.warpAffine(test_img, similarity_M, (w, h))
img_similar_rgb = cv2.cvtColor(img_similar, cv2.COLOR_BGR2RGB)

cv2.imwrite("results/01相似变换.png", img_similar)

# ---------- 2.2 仿射变换
src_aff = np.float32([[50,50], [w-50,50], [50,h-50]])
dst_aff = np.float32([[70,80], [w-30,60], [40,h-40]])
affine_M = cv2.getAffineTransform(src_aff, dst_aff)
img_affine = cv2.warpAffine(test_img, affine_M, (w, h))
img_affine_rgb = cv2.cvtColor(img_affine, cv2.COLOR_BGR2RGB)

cv2.imwrite("results/02仿射变换.png", img_affine)

# ---------- 2.3 透视变换
src_pers = np.float32([[50,50], [w-50,50], [w-50,h-50], [50,h-50]])
dst_pers = np.float32([[80,90], [w-30,70], [w-40,h-30], [30,h-60]])
pers_M = cv2.getPerspectiveTransform(src_pers, dst_pers)
img_pers = cv2.warpPerspective(test_img, pers_M, (w, h))
img_pers_rgb = cv2.cvtColor(img_pers, cv2.COLOR_BGR2RGB)

cv2.imwrite("results/03透视变换.png", img_pers)

# ===================== 3. 三种变换结果对比总图 =====================
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes[0,0].imshow(test_img_rgb)
axes[0,0].set_title("original", fontsize=14)
axes[0,0].axis('off')

axes[0,1].imshow(img_similar_rgb)
axes[0,1].set_title("similarity", fontsize=14)
axes[0,1].axis('off')

axes[1,0].imshow(img_affine_rgb)
axes[1,0].set_title("affine", fontsize=14)
axes[1,0].axis('off')

axes[1,1].imshow(img_pers_rgb)
axes[1,1].set_title("perspective", fontsize=14)
axes[1,1].axis('off')

plt.tight_layout()

plt.savefig("results/00三类变换总对比.png", dpi=300, bbox_inches='tight')
plt.close()

# ===================== 4. 透视畸变图像校正 =====================
src_distort = np.float32([
    [120, 150],   # 左上
    [900, 130],   # 右上
    [950, 850],   # 右下
    [80, 880]     # 左下
])
# 标准A4宽高比
dst_correct = np.float32([
    [0, 0],
    [800, 0],
    [800, 1130],
    [0, 1130]
])

correct_M = cv2.getPerspectiveTransform(src_distort, dst_correct)
img_correct = cv2.warpPerspective(distort_img, correct_M, (800, 1130))
img_correct_rgb = cv2.cvtColor(img_correct, cv2.COLOR_BGR2RGB)

cv2.imwrite("results/04透视校正结果.png", img_correct)

# 畸变原图&校正图对比
fig2, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].imshow(distort_img_rgb)
ax[0].set_title("distorted", fontsize=14)
ax[0].axis('off')

ax[1].imshow(img_correct_rgb)
ax[1].set_title("corrected", fontsize=14)
ax[1].axis('off')

plt.tight_layout()

plt.savefig("results/05畸变与校正对比.png", dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ 所有实验运行完成！results文件夹内为生成的图片")