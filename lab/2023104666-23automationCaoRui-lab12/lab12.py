import cv2
import numpy as np
import glob
import os

# ==========================
# 创建结果文件夹
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "calibration_images")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULT_DIR, exist_ok=True)

# ==========================
# 棋盘参数
# ==========================
CHECKERBOARD = (9, 6)      # 内角点数量（列，行）
SQUARE_SIZE = 25           # mm

# ==========================
# 亚像素优化参数
# ==========================
criteria = (
    cv2.TERM_CRITERIA_EPS +
    cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

# ==========================
# 世界坐标系中的角点
# ==========================
objp = np.zeros(
    (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
    np.float32
)

objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)

objp *= SQUARE_SIZE

# ==========================
# 保存所有角点
# ==========================
objpoints = []
imgpoints = []

# ==========================
# 读取所有图片
# ==========================
images = []
extensions = ["*.jpg", "*.png", "*.jpeg", "*.bmp"]

for ext in extensions:
    images.extend(
        glob.glob(os.path.join(IMAGE_DIR, ext))
    )

print("找到图片数量：", len(images))

if len(images) == 0:
    print("没有找到标定图片！")
    exit()

# ==========================
# 检测角点
# ==========================
success_count = 0

for idx, fname in enumerate(images):

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        None
    )

    if ret:

        success_count += 1

        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        imgpoints.append(corners2)

        draw_img = img.copy()

        cv2.drawChessboardCorners(
            draw_img,
            CHECKERBOARD,
            corners2,
            ret
        )

        save_name = os.path.join(
            RESULT_DIR,
            f"corners_{success_count}.jpg"
        )

        cv2.imwrite(save_name, draw_img)

        print(f"角点检测成功：{fname}")

    else:
        print(f"角点检测失败：{fname}")

print("成功检测图片数量：", success_count)

if success_count < 10:
    print("成功图片太少，建议重新拍摄！")
    exit()

# ==========================
# 相机标定
# ==========================
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None
)

# ==========================
# 输出结果
# ==========================
print("\n===== 相机内参矩阵 =====")
print(mtx)

print("\n===== 畸变参数 =====")
print(dist)

# ==========================
# 重投影误差
# ==========================
mean_error = 0

for i in range(len(objpoints)):

    imgpoints2, _ = cv2.projectPoints(
        objpoints[i],
        rvecs[i],
        tvecs[i],
        mtx,
        dist
    )

    error = cv2.norm(
        imgpoints[i],
        imgpoints2,
        cv2.NORM_L2
    ) / len(imgpoints2)

    mean_error += error

mean_error /= len(objpoints)

print("\n===== 重投影误差 =====")
print(mean_error)

# ==========================
# 保存标定结果
# ==========================
result_txt = os.path.join(
    RESULT_DIR,
    "calibration_result.txt"
)

with open(result_txt, "w", encoding="utf-8") as f:

    f.write("Camera Matrix:\n")
    f.write(str(mtx))
    f.write("\n\n")

    f.write("Distortion Coefficients:\n")
    f.write(str(dist))
    f.write("\n\n")

    f.write("Mean Reprojection Error:\n")
    f.write(str(mean_error))

print("标定结果已保存")

# ==========================
# 去畸变示例
# ==========================
img = cv2.imread(images[0])

h, w = img.shape[:2]

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
    mtx,
    dist,
    (w, h),
    1,
    (w, h)
)

dst = cv2.undistort(
    img,
    mtx,
    dist,
    None,
    newcameramtx
)

cv2.imwrite(
    os.path.join(
        RESULT_DIR,
        "original.jpg"
    ),
    img
)

cv2.imwrite(
    os.path.join(
        RESULT_DIR,
        "undistorted.jpg"
    ),
    dst
)

print("去畸变图片已保存")

print("\n========================")
print("Camera Calibration 完成")
print("========================")
print("结果保存在：")
print(RESULT_DIR)