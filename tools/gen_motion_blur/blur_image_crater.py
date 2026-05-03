import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条，可选

# -----------------------------
# 1️⃣ 随机抖动模糊函数
# -----------------------------
def generate_small_random_motion_kernel(kernel_size=25, trajectory_points=100):
    trajectory = np.zeros((trajectory_points, 2), dtype=np.float32)
    velocity = np.random.randn(2) * 0.02
    inertia = 0.95
    noise_scale = 0.05

    for i in range(1, trajectory_points):
        velocity = inertia * velocity + np.random.randn(2) * noise_scale
        trajectory[i] = trajectory[i - 1] + velocity

    trajectory -= trajectory.mean(axis=0)
    trajectory /= np.max(np.abs(trajectory))
    trajectory = trajectory * (kernel_size / 4) + kernel_size / 2
    trajectory = np.clip(np.round(trajectory).astype(int), 0, kernel_size - 1)

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    for (x, y) in trajectory:
        kernel[y, x] += 1
    kernel /= np.sum(kernel)
    return kernel

def apply_small_motion_blur(image, kernel_size=25, trajectory_points=100):
    kernel = generate_small_random_motion_kernel(kernel_size, trajectory_points)
    blurred = cv2.filter2D(image, -1, kernel)
    return np.clip(blurred, 0, 255).astype(np.uint8), kernel

# -----------------------------
# 2️⃣ 批量处理函数
# -----------------------------
def batch_blur_images(input_folders, output_folders, kernel_size=25, trajectory_points=120):
    for in_folder, out_folder in zip(input_folders, output_folders):
        os.makedirs(out_folder, exist_ok=True)

        # 遍历目录
        for root, _, files in os.walk(in_folder):
            rel_path = os.path.relpath(root, in_folder)
            save_root = os.path.join(out_folder, rel_path)
            os.makedirs(save_root, exist_ok=True)

            for file_name in tqdm(files):
                if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    continue

                img_path = os.path.join(root, file_name)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"⚠️ 无法读取: {img_path}")
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_blur, _ = apply_small_motion_blur(img_rgb, kernel_size, trajectory_points)
                img_blur_bgr = cv2.cvtColor(img_blur, cv2.COLOR_RGB2BGR)

                save_path = os.path.join(save_root, file_name)
                cv2.imwrite(save_path, img_blur_bgr)

# -----------------------------
# 3️⃣ 执行批量处理
# -----------------------------
input_folders = [
    '../datasets/GeoMartian/Geo-train/images',
    '../datasets/GeoMartian/Geo-val/images',
    '../datasets/GeoMartian/Geo-test-dev/images',
]

output_folders = [
    '../datasets/GeoMartian_blur/2_DeblurGAN/blur_image/GeoMartian-train/images',
    '../datasets/GeoMartian_blur/2_DeblurGAN/blur_image/GeoMartian-val/images',
    '../datasets/GeoMartian_blur/2_DeblurGAN/blur_image/GeoMartian-test-dev/images',
]

batch_blur_images(input_folders, output_folders, kernel_size=25, trajectory_points=120)
print("✅ 批量运动模糊处理完成！")
