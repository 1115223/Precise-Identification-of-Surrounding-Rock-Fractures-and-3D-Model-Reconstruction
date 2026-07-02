# -*- coding: utf-8 -*-

import os
import zipfile
import shutil
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from skimage import io as skio
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_otsu, threshold_li
from skimage.filters.ridges import frangi, sato
from skimage.morphology import remove_small_objects, skeletonize, binary_opening, binary_closing, rectangle
from skimage.measure import label, regionprops
from skimage.exposure import rescale_intensity
from skimage.util import img_as_ubyte

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


ZIP_PATH = r"附件1.zip"
WORKDIR  = r"问题1数据集"
OUT_DIR  = r"问题1结果"

for d in [WORKDIR, OUT_DIR]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

if os.path.exists(ZIP_PATH):
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
            zf.extractall(WORKDIR)
        print(f"文件 {ZIP_PATH} 已经成功解压到 {WORKDIR}")
    except Exception as e:
        print(f"解压时出错：{e}")
else:
    print(f"没有找到压缩文件 {ZIP_PATH}，请确认是否放在目录下或直接放图片到 {WORKDIR}")

# 收集所有图像文件
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
def list_images(root):
    fs = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                fs.append(os.path.join(r, f))
    return sorted(fs)

img_files = list_images(WORKDIR)
print(f"本次检测到 {len(img_files)} 张图像：")
for i, p in enumerate(img_files[:20], start=1):
    print(f"  {i}. {os.path.relpath(p, WORKDIR)}")
if len(img_files) > 20:
    print(f"  ... 共 {len(img_files)} 张，仅展示前20张")


def get_filename(path):
    return os.path.splitext(os.path.basename(path))[0]


# --------------------- 核心函数 ---------------------

def remove_stripe(img):
    col_med = np.median(img, axis=0, keepdims=True)
    baseline = col_med - np.median(col_med)
    out = img - baseline
    out = rescale_intensity(out, in_range="image", out_range=(0,1))
    return out

def highpass_filter(img):
    g1 = gaussian(img, sigma=(1.0, 10.0))
    g2 = gaussian(img, sigma=(1.0, 3.0))
    out = img - g1 + (g2 - g1)*0.5
    out = rescale_intensity(out, in_range="image", out_range=(0,1))
    return out

def crack_map(img):
    inv = 1.0 - img
    f = frangi(inv, sigmas=np.linspace(1.0, 4.0, 8), black_ridges=True)
    s = sato(inv,   sigmas=np.linspace(1.0, 4.0, 8), black_ridges=True)
    cr = np.maximum(f, s)
    cr = rescale_intensity(cr, in_range="image", out_range=(0,1))
    return cr

def remove_vertical(mask, min_width_px=6, vertical_tol_deg=12):
    lab = label(mask)
    props = regionprops(lab)
    remove_labels = set()
    for p in props:
        minr, minc, maxr, maxc = p.bbox
        h = maxr - minr
        w = maxc - minc
        ori = p.orientation
        deg_from_vertical = abs(90.0 - abs(np.degrees(ori)))
        if w >= min_width_px and h > 0.6*mask.shape[0] and deg_from_vertical <= vertical_tol_deg:
            remove_labels.add(p.label)
    if remove_labels:
        mask = mask.copy()
        for lb in remove_labels:
            mask[lab == lb] = False
    return mask

def refine_mask(mask):
    mask = binary_opening(mask, rectangle(3,3))
    mask = remove_small_objects(mask, min_size=100)
    mask = binary_closing(mask, rectangle(3,3))
    return mask

def measure_components(skeleton):
    lab = label(skeleton)
    props = regionprops(lab)
    lengths, orients = [], []
    for p in props:
        lengths.append(p.area)
        orients.append(p.orientation)
    if not lengths:
        return dict(num_components=0,total_length_px=0,avg_component_length_px=0.0,
                    orientation_entropy=0.0,mean_orientation_deg=np.nan)
    total_len = int(np.sum(lengths))
    avg_len   = float(np.mean(lengths))
    bins = np.linspace(-np.pi/2, np.pi/2, 19)
    hist, _ = np.histogram(orients, bins=bins, density=True)
    hist = hist + 1e-9
    ent  = -np.sum(hist * np.log(hist)) / np.log(len(hist))
    mean_ori = np.degrees(np.arctan2(np.mean(np.sin(orients)), np.mean(np.cos(orients))))
    return dict(num_components=len(lengths), total_length_px=int(total_len),
                avg_component_length_px=float(avg_len),
                orientation_entropy=float(ent), mean_orientation_deg=float(mean_ori))

def process_image(img):
    if img.ndim == 3:
        gray = rgb2gray(img)
    else:
        gray = img.astype(np.float32)
        if gray.max() > 1.0:
            gray = gray / 255.0
    gray = rescale_intensity(gray, in_range="image", out_range=(0,1))

    before = np.var(np.mean(gray, axis=0))
    ds = remove_stripe(gray)
    after  = np.var(np.mean(ds, axis=0))
    destripe_reduction = (before - after) / (before + 1e-8)
    hp = highpass_filter(ds)
    crackness = crack_map(hp)

    thr_o = threshold_otsu(crackness)
    try:
        thr_l = threshold_li(crackness)
        thr = min(thr_o, thr_l)
    except Exception:
        thr = thr_o * 0.95
    mask = crackness > thr
    mask = remove_vertical(mask)
    mask = refine_mask(mask)
    skel = skeletonize(mask)
    metrics = measure_components(skel)
    crack_area_ratio = float(mask.sum() / mask.size)

    return dict(gray=gray, destriped=ds, highpass=hp, crackness=crackness,
                mask=mask, skeleton=skel,
                destripe_reduction=float(destripe_reduction),
                crack_area_ratio=crack_area_ratio, **metrics)


def save_mask(mask_bool, path_png):
    out = np.ones(mask_bool.shape, dtype=np.uint8)*255
    out[mask_bool] = 0
    skio.imsave(path_png, out)


# --------------------- 批量处理 ---------------------

records_all = []
for p in img_files:
    try:
        img = skio.imread(p)
    except Exception as e:
        print(f"图片 {p} 打不开，报错：{e}")
        continue
    res = process_image(img)

    basename = get_filename(p)
    mask_path = os.path.join(OUT_DIR, f"{basename}__mask.png")
    save_mask(res["mask"], mask_path)

    crack_path = os.path.join(OUT_DIR, f"{basename}__crackmap.png")
    skio.imsave(crack_path, img_as_ubyte(rescale_intensity(res["crackness"], in_range=(0,1), out_range=(0,1))))

    overlay_path = os.path.join(OUT_DIR, f"{basename}__skeleton.png")
    fig = plt.figure(figsize=(6,10))
    plt.imshow(res["gray"])
    yx = np.argwhere(res["skeleton"])
    if yx.size:
        plt.plot(yx[:,1], yx[:,0], '.', markersize=0.5)
    plt.axis('off')
    plt.savefig(overlay_path, bbox_inches="tight", dpi=220)
    plt.close(fig)

    H, W = res["mask"].shape
    records_all.append(dict(
        ImageName=os.path.relpath(p, WORKDIR),
        Height=H, Width=W,
        CrackAreaRatio=res["crack_area_ratio"],
        StripeReduction=res["destripe_reduction"],
        NumComponents=res["num_components"],
        TotalSkeletonLength=res["total_length_px"],
        AvgComponentLength=res["avg_component_length_px"],
        OrientationEntropy=res["orientation_entropy"],
        MeanOrientationDeg=res["mean_orientation_deg"]
    ))
    print(f"图像 {os.path.relpath(p, WORKDIR)} 处理完成，结果存储在 {OUT_DIR}")

# 保存指标 CSV
df_all = pd.DataFrame(records_all)
csv_all = os.path.join(OUT_DIR, "metrics_all.csv")
df_all.to_csv(csv_all, index=False, encoding="utf-8-sig")
print(f"指标数据已保存到：{csv_all}")

# 打印部分结果
print("\n===== 问题1：全部图像指标（前5条） =====")
if not df_all.empty:
    print(df_all.head().to_string(index=False))
else:
    print("没有生成任何指标数据。")

print("\n所有处理任务已完成，结果请查看文件夹：问题1结果")
