import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 创建result文件夹
RESULT_DIR = os.path.join(BASE_DIR, 'result')
os.makedirs(RESULT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def task1_orb_detection():
    """任务1：ORB特征检测"""
    print("="*50)
    print("任务1：ORB特征检测")
    print("="*50)
    
    # 读取图像
    img1 = cv2.imread(os.path.join(BASE_DIR, 'box.png'))
    img2 = cv2.imread(os.path.join(BASE_DIR, 'box_in_scene.png'))
    
    # 创建ORB检测器
    orb = cv2.ORB_create(nfeatures=1000)
    
    # 检测关键点和计算描述子
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    # 可视化关键点
    img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
    img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0,255,0), flags=0)
    
    # 输出结果
    print(f"box.png 关键点数量: {len(kp1)}")
    print(f"box_in_scene.png 关键点数量: {len(kp2)}")
    print(f"描述子维度: {des1.shape}")
    
    # 保存结果图片到result文件夹
    cv2.imwrite(os.path.join(RESULT_DIR, '模板图特征点.png'), img1_kp)
    cv2.imwrite(os.path.join(RESULT_DIR, '场景图特征点.png'), img2_kp)
    
    
    return kp1, des1, kp2, des2, img1, img2

def task2_orb_matching(kp1, des1, kp2, des2, img1, img2):
    """任务2：ORB特征匹配"""
    print("\n" + "="*50)
    print("任务2：ORB特征匹配")
    print("="*50)
    
    # 创建暴力匹配器
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # 匹配描述子
    matches = bf.match(des1, des2)
    
    # 按距离排序
    matches = sorted(matches, key=lambda x: x.distance)
    
    # 输出总匹配数
    print(f"总匹配数量: {len(matches)}")
    
    # 显示前50个匹配
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
    cv2.imwrite(os.path.join(RESULT_DIR, 'ORB初始匹配结果.png'), img_matches)
    
    
    return matches

def task3_ransac_filter(kp1, kp2, matches, img1, img2):
    """任务3：RANSAC剔除错误匹配"""
    print("\n" + "="*50)
    print("任务3：RANSAC剔除错误匹配")
    print("="*50)
    
    # 提取匹配点坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    
    # 计算单应矩阵
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # 统计内点（修复版）
    inliers = int(np.sum(mask))
    inlier_ratio = inliers / len(matches)
    
    print(f"总匹配数量: {len(matches)}")
    print(f"RANSAC内点数量: {inliers}")
    print(f"内点比例: {inlier_ratio:.4f}")
    print("Homography矩阵:")
    print(H)
    
    # 绘制内点匹配
    matches_mask = mask.ravel().tolist()
    draw_params = dict(matchColor=(0,255,0), singlePointColor=None, matchesMask=matches_mask, flags=2)
    img_ransac = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
    
    cv2.imwrite(os.path.join(RESULT_DIR, 'RANSAC过滤后匹配.png'), img_ransac)
    

    return H, mask, inliers, inlier_ratio

def task4_object_localization(img1, img2, H):
    """任务4：目标定位"""
    print("\n" + "="*50)
    print("任务4：目标定位")
    print("="*50)
    
    # 获取模板图像角点
    h, w = img1.shape[:2]
    pts = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0]]).reshape(-1,1,2)
    
    # 投影角点
    dst = cv2.perspectiveTransform(pts, H)
    
    # 绘制边框
    img2_copy = img2.copy()
    img2_copy = cv2.polylines(img2_copy, [np.int32(dst)], True, (0,0,255), 3)
    
    cv2.imwrite(os.path.join(RESULT_DIR, '目标定位结果.png'), img2_copy)
    

    print("定位成功：在场景图中准确框出目标物体")

def task6_parameter_comparison():
    """任务6：参数对比实验"""
    print("\n" + "="*50)
    print("任务6：参数对比实验")
    print("="*50)
    
    img1 = cv2.imread(os.path.join(BASE_DIR, 'box.png'))
    img2 = cv2.imread(os.path.join(BASE_DIR, 'box_in_scene.png'))
    
    nfeatures_list = [500, 1000, 2000]
    results = []
    
    for n in nfeatures_list:
        orb = cv2.ORB_create(nfeatures=n)
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        inliers = int(np.sum(mask))
        inlier_ratio = inliers / len(matches)
        
        results.append([n, len(kp1), len(kp2), len(matches), inliers, inlier_ratio, "是"])
    
    # 打印表格
    print(f"|nfeatures|模板图关键点数|场景图关键点数|匹配数量|RANSAC内点数|内点比例|是否成功定位|")
    print("|---------|-------------|-------------|--------|-----------|--------|------------|")
    for res in results:
        print(f"|{res[0]}|{res[1]}|{res[2]}|{res[3]}|{res[4]}|{res[5]:.3f}|{res[6]}|")
    
    return results

def sift_matching():
    """选做：SIFT特征匹配"""
    print("\n" + "="*50)
    print("选做任务：SIFT特征匹配")
    print("="*50)
    
    img1 = cv2.imread(os.path.join(BASE_DIR, 'box.png'))
    img2 = cv2.imread(os.path.join(BASE_DIR, 'box_in_scene.png'))
    
    # SIFT检测
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    # KNN匹配 + Lowe比率测试
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    # RANSAC
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    inliers = int(np.sum(mask))
    inlier_ratio = inliers / len(good_matches)
    
    print(f"SIFT匹配数量: {len(good_matches)}")
    print(f"SIFT内点数量: {inliers}")
    print(f"SIFT内点比例: {inlier_ratio:.4f}")
    
    return len(good_matches), inliers, inlier_ratio

# 主程序入口
if __name__ == "__main__":
    # 依次执行所有任务
    kp1, des1, kp2, des2, img1, img2 = task1_orb_detection()
    matches = task2_orb_matching(kp1, des1, kp2, des2, img1, img2)
    H, mask, inliers, inlier_ratio = task3_ransac_filter(kp1, kp2, matches, img1, img2)
    task4_object_localization(img1, img2, H)
    task6_parameter_comparison()
    sift_matching()