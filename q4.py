# -*- coding: utf-8 -*-
"""
问题4 完整脚本（彩色裂隙叠加 + 可视化 + 中文输出文件名）
说明：
 - 确保当前目录有附件4.zip 或者已将图像放置到 问题4数据 下
 - 运行后会生成：
    问题4结果/输出/*.csv, 问题4结果/图件彩色/*.png
    问题4结果/输出/问题4_摘要.json
"""
import os, re, zipfile, shutil, math, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io as skio
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, sobel
from skimage.exposure import rescale_intensity, equalize_adapthist
from skimage.morphology import binary_opening, binary_closing, remove_small_objects, square
from skimage.measure import label, regionprops, find_contours
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ---------------- 目录与文件 ----------------
ZIP_FILE = r"附件4.zip"
DATA_DIR = r"问题4数据"
RESULT_DIR = r"问题4结果"
FIG_DIR = os.path.join(RESULT_DIR, "图件彩色")
OUTPUT = os.path.join(RESULT_DIR, "输出")

for d in (DATA_DIR, FIG_DIR, OUTPUT):
    if os.path.exists(d): shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

if os.path.exists(ZIP_FILE):
    with zipfile.ZipFile(ZIP_FILE, "r") as zf:
        zf.extractall(DATA_DIR)
    print(f"已解压：{ZIP_FILE} -> {DATA_DIR}")
else:
    print(f"未找到 {ZIP_FILE} ，请确认路径或将图片放入：{DATA_DIR}")

# ---------------- 钻孔信息 ----------------
borehole_info = pd.DataFrame([
    {"id": "1#", "x": 500, "y": 2000, "depth": 7000},
    {"id": "2#", "x": 1500, "y": 2000, "depth": 7000},
    {"id": "3#", "x": 2500, "y": 2000, "depth": 7000},
    {"id": "4#", "x": 500, "y": 1000, "depth": 5000},
    {"id": "5#", "x": 1500, "y": 1000, "depth": 7000},
    {"id": "6#", "x": 2500, "y": 1000, "depth": 7000},
])

circ_mm = math.pi * 30.0
radius_mm = circ_mm / (2 * math.pi)

EXTS = (".png",".jpg",".jpeg",".bmp",".tif",".tiff")

# ---------------- 工具函数 ----------------
def scan_images(root_dir):
    imgs=[]
    for r,_,fs in os.walk(root_dir):
        for f in fs:
            if f.lower().endswith(EXTS):
                imgs.append(os.path.join(r,f))
    return sorted(imgs)

def gray_preprocess(img):
    if img.ndim==3: g = rgb2gray(img)
    else: g = img.astype(float)
    if g.max() > 1: g = g/255.0
    g = rescale_intensity(g, out_range=(0,1))
    g = equalize_adapthist(g, clip_limit=0.01)
    return g

def crack_segment(gray_img):
    edge_map = sobel(gray_img)
    thresh = threshold_otsu(gray_img)
    mask_intensity = gray_img < thresh
    mask = (edge_map > np.percentile(edge_map, 70)) | mask_intensity
    mask = binary_opening(mask, square(3))
    mask = binary_closing(mask, square(3))
    mask = remove_small_objects(mask, 120)
    return mask

def extract_main_contour(bin_mask):
    if bin_mask.sum() < 50: return None
    lbl_img = label(bin_mask)
    props = regionprops(lbl_img)
    if not props: return None
    props.sort(key=lambda r: r.area, reverse=True)
    comp = (lbl_img == props[0].label).astype(float)
    contours = find_contours(comp, 0.5)
    if not contours: return None
    contours.sort(key=lambda c: len(c), reverse=True)
    c = contours[0]
    return c[:,1], c[:,0]

def compute_pca(u_pts, v_pts):
    pts = np.vstack([u_pts, v_pts]).T
    mu = pts.mean(axis=0)
    q = pts - mu
    cov = np.cov(q.T)
    w, V = np.linalg.eigh(cov)
    e1 = V[:,1]; e2 = V[:,0]
    E = np.vstack([e1,e2]).T
    uv = q @ E
    return uv[:,0], uv[:,1], E, mu

def equal_sample(u,v,N=200):
    idx = np.argsort(u); u=u[idx]; v=v[idx]
    if (u[-1]-u[0]) < 1e-6:
        s = np.r_[0, np.cumsum(np.hypot(np.diff(u), np.diff(v)))]
        ss = np.linspace(0, s[-1], N+1)
        uu = np.interp(ss, s, u); vv = np.interp(ss, s, v)
        return uu, vv
    us = np.linspace(u[0], u[-1], N+1)
    vs = np.interp(us, u, v)
    return us, vs

def compute_z2(u, v):
    du = np.diff(u); dv = np.diff(v)
    du = np.where(np.abs(du)<1e-9, np.sign(du)*1e-9 + (du==0)*1e-9, du)
    return float(np.sqrt(np.mean((dv/du)**2)))

def z2_to_jrc(z2_val):
    return 51.85*(z2_val**0.6) - 10.37

def sine_fit(u, v):
    u0,u1 = float(u.min()), float(u.max())
    P = max(u1-u0, 10.0)
    w = 2*np.pi/P
    S = np.sin(w*u); C = np.cos(w*u)
    A = np.vstack([S, C, np.ones_like(u)]).T
    sol, *_ = np.linalg.lstsq(A, v, rcond=None)
    a,b,c = sol
    R = float(np.hypot(a,b))
    beta = float(np.arctan2(b, a))
    yhat = a*S + b*C + c
    r2_val = 1 - np.sum((v-yhat)**2)/(np.sum((v-v.mean())**2)+1e-12)
    return dict(R=R, P=P, beta=beta, C=float(c), r2=r2_val)

def plane_from_sine(sf, img_shape, depth, radius_mm):
    H, W = img_shape
    mm_per_px = depth / float(H)
    R_mm = sf["R"] * mm_per_px
    dip = math.atan2(max(R_mm,0.0), radius_mm)
    az = sf["beta"]
    C_mm = sf["C"] * mm_per_px
    hz = 1.0 / max(math.tan(dip), 1e-6)
    nx = math.cos(az); ny = math.sin(az)
    n = np.array([nx, ny, hz], dtype=float)
    n = n / np.linalg.norm(n)
    return dict(R_mm=R_mm, dip=dip, az=az, C_mm=C_mm, normal=n)

def detect_hole(fname):
    base = os.path.basename(fname)
    m = re.search(r'([1-6])#', base)
    if m: return f"{m.group(1)}#"
    m = re.search(r'(?:BH|孔|Hole|H)([1-6])', base, re.IGNORECASE)
    if m: return f"{m.group(1)}#"
    parts = os.path.normpath(fname).split(os.sep)
    for p in parts:
        m = re.search(r'([1-6])#', p)
        if m: return f"{m.group(1)}#"
    return None

def plane_z_at_xy(n_vec, p0, xy):
    nx, ny, nz = n_vec
    if abs(nz) < 1e-9: return None
    return p0[2] - (nx*(xy[0]-p0[0]) + ny*(xy[1]-p0[1]))/nz

def overlay_crack(img, mask):
    """彩色裂隙叠加图"""
    if img.ndim==2: img_color = np.stack([img]*3, axis=-1)
    else: img_color = img.copy()
    img_color = img_color.astype(float)/img_color.max()
    img_color[mask>0] = [1,0,0]  # 红色裂隙
    return img_color

# ---------------- 主处理 ----------------
img_list = scan_images(DATA_DIR)
print(f"找到 {len(img_list)} 张图片（目录：{DATA_DIR}）")

hole_map = {}
holes_cycle = borehole_info["id"].tolist()
k = 0
for path in img_list:
    h = detect_hole(path)
    if h is None:
        h = holes_cycle[k % len(holes_cycle)]
        k += 1
    hole_map[path] = h

records = []
for path in img_list:
    try: img = skio.imread(path)
    except Exception:
        print(f"读取失败：{path}，跳过")
        continue
    gray = gray_preprocess(img)
    H, W = gray.shape
    hole_id = hole_map.get(path, "1#")
    depth_mm = float(borehole_info[borehole_info["id"]==hole_id]["depth"].values[0])
    mask = crack_segment(gray)
    xy = extract_main_contour(mask)
    if xy is None:
        print(f"未在图像中检测到连贯裂隙：{os.path.basename(path)}，跳过")
        continue
    xs, ys = xy
    u, v, E, mu = compute_pca(xs, ys)
    us, vs = equal_sample(u, v, 200)
    z2 = compute_z2(us, vs)
    jrc = z2_to_jrc(z2)
    sf = sine_fit(us, vs)
    plane = plane_from_sine(sf, gray.shape, depth_mm, radius_mm)
    bh = borehole_info[borehole_info["id"]==hole_id].iloc[0]
    rec = dict(
        image=os.path.basename(path),
        hole=hole_id,
        H=int(H), W=int(W),
        depth_mm=depth_mm,
        z2=z2,
        JRC=jrc,
        **sf, **plane,
        bh_x=bh["x"], bh_y=bh["y"],
        p0x=bh["x"], p0y=bh["y"], p0z=plane["C_mm"]
    )
    records.append(rec)
    print(f"已处理：{os.path.basename(path)} → 孔号 {hole_id}，JRC={jrc:.2f}")

    # 保存彩色裂隙叠加图
    overlay = overlay_crack(img, mask)
    out_overlay = os.path.join(FIG_DIR, f"{hole_id}_{os.path.basename(path)}")
    plt.imsave(out_overlay, overlay)

df_rec = pd.DataFrame(records)

# ---------------- 导出 CSV ----------------
out_csv1 = os.path.join(OUTPUT, "问题4_裂隙平面.csv")
df_rec.to_csv(out_csv1, index=False, encoding='utf-8-sig')
print(f"已保存：{out_csv1}")

# ---------------- 可视化 ----------------
if len(df_rec):
    # 饼图
    counts = df_rec["hole"].value_counts()
    fig, ax = plt.subplots(figsize=(7,7))
    ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2"))
    ax.set_title("问题4：钻孔裂隙数量占比（饼图）")
    out_pie = os.path.join(FIG_DIR, "问题4_钻孔饼图.png")
    fig.savefig(out_pie, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存可视化：{out_pie}")

# ---------------- 输出摘要 ----------------
summary = dict(
    n_images=len(img_list),
    n_detected=len(df_rec)
)
with open(os.path.join(OUTPUT,"问题4_摘要.json"),"w",encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("问题4处理完成，摘要已保存。")
