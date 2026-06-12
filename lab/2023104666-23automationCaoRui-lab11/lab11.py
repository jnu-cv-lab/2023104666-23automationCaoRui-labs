import os
import cv2
import json
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import train_test_split

# =========================
# 路径配置
# =========================

DATASET_DIR = "/mnt/d/下载/archive"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULT_DIR = os.path.join(BASE_DIR, "results")
SKELETON_DIR = os.path.join(BASE_DIR, "skeleton_data")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(SKELETON_DIR, exist_ok=True)

# =========================
# 参数
# =========================

TARGET_FRAMES = 30
NUM_KEYPOINTS = 33
FEATURE_DIM = 132

# =========================
# 类别映射
# =========================

label_map = {
    "forehand_drive": 0,
    "forehand_lift": 1,
    "forehand_net_shot": 2,
    "forehand_clear": 3,
    "backhand_drive": 4,
    "backhand_net_shot": 5
}

# 保存label_map

with open(
    os.path.join(SKELETON_DIR, "label_map.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(label_map, f, indent=4)

# =========================
# MediaPipe
# =========================

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# 重采样
# =========================

def resample_sequence(sequence, target_len=30):

    sequence = np.array(sequence)

    if len(sequence) == 0:
        return np.zeros((target_len, FEATURE_DIM))

    old_idx = np.linspace(
        0,
        len(sequence)-1,
        len(sequence)
    )

    new_idx = np.linspace(
        0,
        len(sequence)-1,
        target_len
    )

    result = []

    for d in range(sequence.shape[1]):
        result.append(
            np.interp(
                new_idx,
                old_idx,
                sequence[:, d]
            )
        )

    result = np.array(result).T

    return result

# =========================
# 简单归一化
# =========================

def normalize_pose(frame_feature):

    pts = frame_feature.reshape(33,4)

    left_hip = pts[23,:2]
    right_hip = pts[24,:2]

    center = (left_hip + right_hip)/2

    pts[:,0] -= center[0]
    pts[:,1] -= center[1]

    left_shoulder = pts[11,:2]
    right_shoulder = pts[12,:2]

    shoulder_width = np.linalg.norm(
        left_shoulder-right_shoulder
    )

    if shoulder_width > 1e-6:
        pts[:,:3] /= shoulder_width

    return pts.flatten()

# =========================
# 视频转骨架
# =========================

def extract_skeleton(video_path):

    cap = cv2.VideoCapture(video_path)

    sequence = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        result = pose.process(rgb)

        if result.pose_landmarks:

            feat = []

            for lm in result.pose_landmarks.landmark:

                feat.extend([
                    lm.x,
                    lm.y,
                    lm.z,
                    lm.visibility
                ])

            feat = np.array(feat)

            feat = normalize_pose(feat)

            sequence.append(feat)

    cap.release()

    sequence = resample_sequence(
        sequence,
        TARGET_FRAMES
    )

    return sequence

# =========================
# 骨架可视化
# =========================

def save_sample_skeleton():

    sample_class = list(label_map.keys())[0]

    sample_video = os.path.join(
        DATASET_DIR,
        sample_class,
        sorted(os.listdir(
            os.path.join(DATASET_DIR,sample_class)
        ))[0]
    )

    cap = cv2.VideoCapture(sample_video)

    ret, frame = cap.read()

    cap.release()

    if not ret:
        return

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    result = pose.process(rgb)

    if result.pose_landmarks:

        plt.figure(figsize=(6,6))

        xs = []
        ys = []

        for lm in result.pose_landmarks.landmark:
            xs.append(lm.x)
            ys.append(-lm.y)

        plt.scatter(xs,ys)

        plt.title("Sample Skeleton")

        plt.savefig(
            os.path.join(
                RESULT_DIR,
                "sample_skeleton.png"
            )
        )

        plt.close()

        print("骨架示例图已保存")

# =========================
# 主程序
# =========================

all_X = []
all_y = []

print("="*50)
print("开始提取骨架序列")
print("="*50)

for class_name,label in label_map.items():

    class_dir = os.path.join(
        DATASET_DIR,
        class_name
    )

    if not os.path.exists(class_dir):
        print(f"未找到目录: {class_dir}")
        continue

    videos = sorted(os.listdir(class_dir))

    print(f"\n处理类别: {class_name}")
    print(f"视频数量: {len(videos)}")

    for video_name in tqdm(videos):

        video_path = os.path.join(
            class_dir,
            video_name
        )

        try:

            seq = extract_skeleton(video_path)

            all_X.append(seq)
            all_y.append(label)

        except Exception as e:

            print(
                f"跳过 {video_name}: {e}"
            )

all_X = np.array(all_X)
all_y = np.array(all_y)

print("\n数据集形状:")
print(all_X.shape)
print(all_y.shape)

# =========================
# 划分数据集
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    all_X,
    all_y,
    test_size=0.2,
    random_state=42,
    stratify=all_y
)

# =========================
# 保存
# =========================

np.save(
    os.path.join(
        SKELETON_DIR,
        "X_train.npy"
    ),
    X_train
)

np.save(
    os.path.join(
        SKELETON_DIR,
        "y_train.npy"
    ),
    y_train
)

np.save(
    os.path.join(
        SKELETON_DIR,
        "X_test.npy"
    ),
    X_test
)

np.save(
    os.path.join(
        SKELETON_DIR,
        "y_test.npy"
    ),
    y_test
)

print("\n保存完成")

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

save_sample_skeleton()

print("\n预处理完成")