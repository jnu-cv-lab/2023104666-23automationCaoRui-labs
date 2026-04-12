import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# -------------------------- 1. 生成测试图 --------------------------
def generate_checkerboard(size=512, block_size=8):
    """生成棋盘格测试图"""
    img = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            if (i // block_size + j // block_size) % 2 == 0:
                img[i, j] = 255
    return img

def generate_chirp(size=512):
    """生成Chirp测试图（频率从0到π线性增加）"""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)
    chirp = np.sin(np.pi * r * size / 2)
    chirp = (chirp - chirp.min()) / (chirp.max() - chirp.min()) * 255
    return chirp.astype(np.uint8)

# -------------------------- 2. 下采样与滤波 --------------------------
def direct_downsample(img, M=4):
    """直接下采样（每隔M个像素取一个）"""
    return img[::M, ::M]

def gaussian_blur(img, sigma=1.8):
    """高斯模糊（抗混叠预滤波）"""
    ksize = int(4 * sigma + 0.5) * 2 + 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

def blur_downsample(img, M=4, sigma=1.8):
    """高斯滤波后下采样"""
    blurred = gaussian_blur(img, sigma)
    return direct_downsample(blurred, M)

# -------------------------- 3. FFT频谱分析 --------------------------
def compute_fft_spectrum(img):
    """计算图像的FFT频谱（中心化）"""
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)
    return magnitude

# -------------------------- 4. 自适应下采样 --------------------------
def compute_gradient_magnitude(img):
    """计算梯度幅值（Sobel算子）"""
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    grad_mag = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)
    return grad_mag

def adaptive_downsample(img, base_M=4, sigma_base=1.8):
    """自适应下采样"""
    h, w = img.shape
    grad_mag = compute_gradient_magnitude(img)
    block_size = base_M
    M_map = np.zeros_like(img, dtype=np.int32)
    sigma_map = np.zeros_like(img, dtype=np.float32)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block_grad = grad_mag[i:i+block_size, j:j+block_size].mean()
            if block_grad > 0.3:
                M = 2
                sigma = 0.45 * M
            else:
                M = 4
                sigma = 0.45 * M
            M_map[i:i+block_size, j:j+block_size] = M
            sigma_map[i:i+block_size, j:j+block_size] = sigma

    blurred = np.zeros_like(img, dtype=np.float32)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            sigma = sigma_map[i, j]
            block = img[i:i+block_size, j:j+block_size]
            blurred_block = gaussian_blur(block, sigma)
            blurred[i:i+block_size, j:j+block_size] = blurred_block
    blurred = blurred.astype(np.uint8)

    downsampled = []
    for i in range(0, h, block_size):
        row = []
        for j in range(0, w, block_size):
            M = M_map[i, j]
            row.append(blurred[i, j])
        downsampled.append(row)
    return np.array(downsampled), M_map


# 创建results文件夹
os.makedirs("results", exist_ok=True)

# -------------------------- 第一部分：混叠现象与抗混叠验证 --------------------------
print("========第一部分：混叠现象与抗混叠验证 ========")

checker = generate_checkerboard(size=512, block_size=8)
chirp = generate_chirp(size=512)
cv2.imwrite("results/checker_original.png", checker)
cv2.imwrite("results/chirp_original.png", chirp)

checker_direct = direct_downsample(checker, M=4)
chirp_direct = direct_downsample(chirp, M=4)
cv2.imwrite("results/checker_direct_downsample.png", checker_direct)
cv2.imwrite("results/chirp_direct_downsample.png", chirp_direct)

checker_blur = blur_downsample(checker, M=4, sigma=1.8)
chirp_blur = blur_downsample(chirp, M=4, sigma=1.8)
cv2.imwrite("results/checker_blur_downsample.png", checker_blur)
cv2.imwrite("results/chirp_blur_downsample.png", chirp_blur)

# -------------------------- 第二部分：σ公式验证 --------------------------
print("\n======== 第二部分：σ公式验证 ========")
M = 4
sigmas = [0.5, 1.0, 2.0, 4.0]
theo_sigma = 0.45 * M

# -------------------------- 第三部分：自适应下采样 --------------------------
print("\n======== 第三部分：自适应下采样 ========")
test_img = np.hstack([checker, chirp])
cv2.imwrite("results/adaptive_test_img.png", test_img)

uniform_down = blur_downsample(test_img, M=4, sigma=1.8)
cv2.imwrite("results/uniform_downsample.png", uniform_down)

adaptive_down, M_map = adaptive_downsample(test_img, base_M=4)
cv2.imwrite("results/adaptive_downsample.png", adaptive_down)

print("\n 结果均在 lab04/results 文件夹下")