# -*- coding: utf-8 -*-
"""
问题2 完整脚本（正弦裂隙拟合 + 彩色叠加 + 可视化 + 中文输出路径）
说明：
 - 需要附件：附件2.zip 或把图片放入 问题2数据集
 - 运行后生成：
    问题2结果/输出/*.csv
    问题2结果/图件彩色/*.png
    问题2结果/摘要.json
"""
import os, zipfile, shutil, warnings, math, json

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io as skio
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_otsu, threshold_li
from skimage.filters.ridges import frangi, sato
from skimage.morphology import remove_small_objects, skeletonize, binary_opening, binary_closing, rectangle
from skimage.exposure import rescale_intensity
from sklearn.cluster import DBSCAN
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

# ---------- Matplotlib 中文设置 ----------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------- 路径设置 ----------------------
zip_file_path = r"附件2.zip"
data_root = r"问题2数据集"
RESULT_DIR = r"问题2结果"
OUTPUT_DIR = os.path.join(RESULT_DIR, "输出")
FIG_DIR = os.path.join(RESULT_DIR, "图件彩色")

for d in [data_root, OUTPUT_DIR, FIG_DIR]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

# ---------------------- 解压数据 ----------------------
if os.path.exists(zip_file_path):
    with zipfile.ZipFile(zip_file_path, "r") as zf:
        zf.extractall(data_root)
    print(f"已解压：{zip_file_path} → {data_root}")
else:
    print(f"未找到压缩包 {zip_file_path}，请确认或将图片直接放入：{data_root}")

# ---------------------- 文件收集 ----------------------
img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def collect_images(root_dir):
    imgs = []
    for r, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(img_exts):
                imgs.append(os.path.join(r, f))
    return sorted(imgs)


image_list = collect_images(data_root)
print(f"找到图片数量：{len(image_list)}（位于：{data_root}）")


# ---------------------- 图像处理与裂隙检测 ----------------------
def remove_stripe_col(img_arr):
    col_median = np.median(img_arr, axis=0, keepdims=True)
    baseline = col_median - np.median(col_median)
    out_img = img_arr - baseline
    return rescale_intensity(out_img, in_range="image", out_range=(0, 1))


def dog_highpass_horizontal(img_arr):
    g1 = gaussian(img_arr, sigma=(1.0, 10.0))
    g2 = gaussian(img_arr, sigma=(1.0, 3.0))
    out_img = img_arr - g1 + (g2 - g1) * 0.5
    return rescale_intensity(out_img, in_range="image", out_range=(0, 1))


def compute_crack_map(img_arr):
    inv_img = 1.0 - img_arr
    f_map = frangi(inv_img, sigmas=np.linspace(1.0, 4.0, 8), black_ridges=True)
    s_map = sato(inv_img, sigmas=np.linspace(1.0, 4.0, 8), black_ridges=True)
    crack_map = np.maximum(f_map, s_map)
    return rescale_intensity(crack_map, in_range="image", out_range=(0, 1))


def segmentation_pipeline(img_arr):
    if img_arr.ndim == 3:
        gray_img = rgb2gray(img_arr)
    else:
        gray_img = img_arr.astype(np.float32)
        if gray_img.max() > 1.0: gray_img /= 255.0
    gray_img = rescale_intensity(gray_img, in_range="image", out_range=(0, 1))
    ds_img = remove_stripe_col(gray_img)
    hp_img = dog_highpass_horizontal(ds_img)
    crack_img = compute_crack_map(hp_img)
    thr_o = threshold_otsu(crack_img)
    try:
        thr_l = threshold_li(crack_img)
        thr_val = min(thr_o, thr_l)
    except Exception:
        thr_val = thr_o * 0.95
    mask_bin = crack_img > thr_val
    mask_bin = binary_opening(mask_bin, rectangle(3, 3))
    mask_bin = remove_small_objects(mask_bin, min_size=80)
    mask_bin = binary_closing(mask_bin, rectangle(3, 3))
    skeleton = skeletonize(mask_bin)
    return gray_img, crack_img, mask_bin, skeleton


def sine_model(x, amplitude, period, phase, offset):
    return amplitude * np.sin(2 * np.pi * x / period + phase) + offset


def ransac_sine_fit(x, y, width_px, trials=60, inlier_thresh=3.0):
    best_fit = None
    n = len(x)
    if n < 40:
        return None
    R0 = (np.percentile(y, 95) - np.percentile(y, 5)) / 2.0
    C0 = np.median(y)
    P0 = max(width_px * 0.8, 10.0)
    bet0 = 0.0
    for _ in range(trials):
        idx = np.random.choice(n, size=min(200, n), replace=False)
        x0, y0 = x[idx], y[idx]
        try:
            popt, _ = curve_fit(sine_model, x0, y0, p0=[R0, P0, bet0, C0], maxfev=20000)
        except Exception:
            continue
        y_pred = sine_model(x, *popt)
        resid = np.abs(y - y_pred)
        inliers = resid < inlier_thresh
        score = int(inliers.sum())
        if (best_fit is None) or (score > best_fit["score"]):
            ss_res = np.sum((y[inliers] - y_pred[inliers]) ** 2)
            ss_tot = np.sum((y[inliers] - np.mean(y[inliers])) ** 2) + 1e-9
            r2_val = 1 - ss_res / ss_tot
            best_fit = dict(params=popt, inliers=inliers, r2=float(r2_val), score=score)
    return best_fit


def overlay_crack(img, mask):
    """彩色裂隙叠加图"""
    if img.ndim == 2:
        img_color = np.stack([img] * 3, axis=-1)
    else:
        img_color = img.copy()
    img_color = img_color.astype(float) / img_color.max()
    img_color[mask > 0] = [1, 0, 0]
    return img_color


# ---------------------- 主处理 ----------------------
all_ransac_results = []
all_single_results = []
panel_paths = []

for img_path in image_list:
    img_name = os.path.basename(img_path)
    print(f"处理中：{img_name} ...")
    img_arr = skio.imread(img_path)
    gray_img, crack_map, mask_bin, skeleton = segmentation_pipeline(img_arr)

    # 保存彩色裂隙叠加
    overlay_img = overlay_crack(gray_img, mask_bin)
    overlay_path = os.path.join(FIG_DIR, f"{os.path.splitext(img_name)[0]}_彩色裂隙.png")
    plt.imsave(overlay_path, overlay_img)

    # 多裂隙拟合
    coords = np.argwhere(skeleton)
    results = []
    if len(coords) > 0:
        XY = np.fliplr(coords)
        clustering = DBSCAN(eps=8.0, min_samples=25).fit(XY)
        labels = clustering.labels_
        for lab_id in sorted(set(labels)):
            if lab_id < 0: continue
            idx = np.where(labels == lab_id)[0]
            if len(idx) < 80: continue
            x = XY[idx, 0].astype(float)
            y = XY[idx, 1].astype(float)
            best = ransac_sine_fit(x, y, gray_img.shape[1], trials=80, inlier_thresh=3.0)
            if best is None: continue
            R_px, P_px, phase, C_px = best["params"]
            results.append(dict(
                image=img_name,
                label=int(lab_id),
                R_px=float(R_px), P_px=float(P_px), beta=float(phase), C_px=float(C_px),
                r2=float(best["r2"]), n_inliers=int(best["inliers"].sum())
            ))
    all_ransac_results.extend(results)

# ---------------------- 保存 CSV ----------------------
df_ransac = pd.DataFrame(all_ransac_results)
csv_path = os.path.join(OUTPUT_DIR, "问题2_RANSAC拟合结果.csv")
df_ransac.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"已保存 CSV：{csv_path}")

# ---------------------- 保存摘要 ----------------------
summary = dict(
    n_images=len(image_list),
    n_ransac=len(all_ransac_results)
)
summary_path = os.path.join(OUTPUT_DIR, "问题2_摘要.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("问题2处理完成，摘要已保存。")
print(summary)
