import cv2
import numpy as np
import matplotlib.pyplot as plt

#调试
print("="*40)
print("✅ 代码开始执行，正在初始化...")
print("="*40)

# -------------------------- 任务1: 使用 OpenCV 读取测试图片 --------------------------
# 替换原来的 img_path 这一行
img_path = "/home/cedric/cv-course/lab01/test1.jpg"
img = cv2.imread(img_path)

# 检查图片是否读取成功
if img is None:
    print(f"错误：无法读取图片 {img_path}")
    print("请检查：")
    print("   1. 图片是否放在 cv-course 文件夹里")
    print("   2. 图片名是否为 test1.jpg（区分大小写）")
    exit()  # 读取失败时明确退出，避免后续报错

# -------------------------- 任务2: 输出图像基本信息--------------------------
height, width, channels = img.shape
print("\n 图像基本信息：")
print(f"   图片路径：{img_path}")
print(f"   尺寸（宽×高）：{width} × {height}")
print(f"   通道数：{channels}（彩色图为3，灰度图为1）")
print(f"   像素数据类型：{img.dtype}")
print("="*40)

# -------------------------- 任务3: 显示原图 --------------------------
print("\n 正在用Matplotlib显示原图...")
plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image (原图)")
plt.axis("off")  # 隐藏坐标轴
plt.show()  # 在VSCode绘图面板显示，无报错

# -------------------------- 任务4: 转换为灰度图并显示 --------------------------
print("\n 正在转换为灰度图...")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(" 正在用Matplotlib显示灰度图...")
plt.figure(figsize=(8, 6))
plt.imshow(gray_img, cmap="gray")  # 灰度图需指定cmap="gray"
plt.title("Grayscale Image (灰度图)")
plt.axis("off")
plt.show()

# -------------------------- 任务5: 保存灰度图 --------------------------
gray_save_path = "gray_test1.jpg"
cv2.imwrite(gray_save_path, gray_img)
print(f"\n 灰度图已保存：{gray_save_path}")

# -------------------------- 任务6: NumPy简单操作 --------------------------
print("\n 开始NumPy像素操作...")
# 1：输出指定坐标像素值（x=100, y=50）
x, y = 100, 50
# 防止坐标超出图片范围（增加容错）
if 0 <= y < height and 0 <= x < width:
    pixel_bgr = img[y, x]
    print(f"   坐标 ({x}, {y}) 的像素值（BGR）：{pixel_bgr}")
else:
    print(f"   坐标 ({x}, {y}) 超出图片范围，跳过像素值输出")

# 2：裁剪图片（适配图片尺寸，避免越界）
crop_size = min(200, width, height)  # 取最小值，防止裁剪超出图片
cropped_img = img[0:crop_size, 0:crop_size]
crop_save_path = "cropped_test1.jpg"
cv2.imwrite(crop_save_path, cropped_img)
print(f"   裁剪后的图片已保存：{crop_save_path}（尺寸：{crop_size}×{crop_size}）")

# -------------------------- 执行完成 --------------------------
print("\n" + "="*40)
print(" 所有任务执行完成")
print(" 生成的文件：")
print(f"   - {gray_save_path}（灰度图）")
print(f"   - {crop_save_path}（裁剪图）")
print("="*40)
