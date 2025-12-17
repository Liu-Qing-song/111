import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def cv_imread(file_path):
    try:
        cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        return cv_img
    except Exception as e:
        return None


def run_paper_based_experiment(image_path):
    print(f"正在处理: {image_path}")
    original = cv_imread(image_path)
    if original is None: return

    # 1. 信号获取
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    plt.figure(figsize=(12, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.suptitle("基于论文方法的信号处理实验：线性 vs 非线性系统", fontsize=16, fontweight='bold')

    # ==========================
    # 阶段一：系统响应对比 (核心教学点)
    # 对应论文：改进滤波算法部分
    # ==========================

    # 系统A: 线性系统 (传统Canny使用的高斯滤波)
    # 这种图冰刺很多，高斯滤波只会把冰刺模糊化，无法去除
    gaussian = cv2.GaussianBlur(gray, (19, 19), 0)

    # 系统B: 非线性系统 (论文提到的中值滤波思路)
    # 针对大冰刺，我们需要大的核 (比如 19x19)
    # 只有非线性系统能把那些尖尖的冰刺“削平”
    median = cv2.medianBlur(gray, 19)

    plt.subplot(2, 3, 1)
    plt.imshow(gaussian, cmap='gray')
    plt.title("1. 线性系统响应 (高斯)\n(冰刺依然存在，边缘模糊)")
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(median, cmap='gray')
    plt.title("2. 非线性系统响应 (中值)\n(冰刺被滤除，边缘保留)")
    plt.axis('off')

    # ==========================
    # 阶段二：特征提取 (差分方程)
    # 对应论文：梯度计算部分
    # ==========================

    # 我们选用效果更好的“非线性系统输出”继续往下做
    input_signal = median

    # 梯度计算 (Sobel算子 - 高通滤波)
    gx = cv2.Sobel(input_signal, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(input_signal, cv2.CV_64F, 0, 1, ksize=3)

    # 梯度幅值
    magnitude = cv2.magnitude(gx, gy)
    # 归一化便于显示
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    plt.subplot(2, 3, 3)
    plt.imshow(magnitude, cmap='jet')
    plt.title("3. 差分系统输出 (梯度幅值)\n(提取出导线主体轮廓)")
    plt.axis('off')

    # ==========================
    # 阶段三：姿态校正 (为了测得准)
    # 这是一个辅助步骤，保证物理意义正确
    # ==========================

    # 简单阈值化梯度图，提取边缘点
    ret, thresh = cv2.threshold(magnitude, 80, 150, cv2.THRESH_BINARY)
    coords = np.column_stack(np.where(thresh > 0))

    rotate_angle = 0
    if len(coords) > 0:
        # 拟合直线找主方向
        [vx, vy, x, y] = cv2.fitLine(coords, cv2.DIST_L2, 0, 0.01, 0.01)
        angle = np.degrees(np.arctan2(vy, vx))[0]-1
        # 导线大约是水平的，算出偏差角度
        if abs(angle) < 45:
            rotate_angle = angle
        else:
            rotate_angle = angle - 90 if angle > 0 else angle + 90

    # 旋转梯度图
    (h, w) = magnitude.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), -rotate_angle, 1.0)
    rotated_mag = cv2.warpAffine(magnitude, M, (w, h))

    plt.subplot(2, 3, 4)
    plt.imshow(rotated_mag, cmap='jet')
    plt.title(f"4. 旋转校正 (修正 {rotate_angle:.1f}°)\n(准备垂直切片)")
    plt.axis('off')

    # ==========================
    # 阶段四：参数估计 (一维波形分析)
    # 对应论文：冰厚计算部分
    # ==========================

    # 自动寻找能量最强的一列（避开背景）
    col_sums = np.sum(rotated_mag, axis=0)
    target_col = np.argmax(col_sums)

    # 提取一维切片
    # 信号1: 原始灰度 (旋转后)
    rotated_img = cv2.warpAffine(gray, M, (w, h))
    signal_slice = rotated_img[:, target_col] / 255.0

    # 信号2: 梯度 (旋转后)
    grad_slice = rotated_mag[:, target_col] / 255.0

    ax = plt.subplot(2, 3, 5)
    ax.plot(signal_slice, 'g-', label='灰度信号 (阶跃)', alpha=0.3)
    ax.plot(grad_slice, 'b-', label='梯度信号 (冲激)', linewidth=2)

    # 寻找梯度峰值 (边缘)
    # 你的图对比度低，梯度峰值可能不明显，我们找最大和次大的峰
    # 或者简单点：找最外侧的两个超过阈值的点
    threshold = 0.15  # 阈值可调
    indices = np.where(grad_slice > threshold)[0]

    if len(indices) > 2:
        top = indices[0]
        bottom = indices[-1]
        thickness = bottom - top

        # 在波形图上标出
        ax.axvline(top, color='r', linestyle='--')
        ax.axvline(bottom, color='r', linestyle='--')
        ax.text((top + bottom) / 2, 0.5, f"Width: {thickness}", color='r', fontweight='bold', ha='center')

        # 在原图上画线展示结果
        result_img = cv2.cvtColor(rotated_img, cv2.COLOR_GRAY2BGR)
        cv2.line(result_img, (0, top), (w, top), (0, 0, 255), 2)
        cv2.line(result_img, (0, bottom), (w, bottom), (0, 0, 255), 2)

        plt.subplot(2, 3, 6)
        plt.imshow(result_img)
        plt.title(f"5. 最终检测结果\n厚度: {thickness} px")
        plt.axis('off')

    ax.set_title("5. 一维波形分析 (1D Slice)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    file_path = r"D:\My_documents\论文撰写\实验设计-代码\line.jpg"
    run_paper_based_experiment(file_path)