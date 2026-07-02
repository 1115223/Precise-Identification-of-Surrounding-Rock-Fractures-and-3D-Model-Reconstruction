# -*- coding: utf-8 -*-
"""
问题3 完整脚本（统一输出到一个文件夹）
"""
import os, zipfile, shutil, math, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io as skio
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.exposure import rescale_intensity, equalize_adapthist
from skimage.morphology import (
    binary_opening, binary_closing, remove_small_objects,
    skeletonize, square, dilation
)
from skimage.measure import label, regionprops
from scipy.optimize import curve_fit
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 路径与解压（统一输出） =====================
ZIP = r"附件3.zip"
ROOT = r"问题3数据集"
OUT  = r"问题3输出汇总"   # 所有输出统一到这里
for d in [ROOT, OUT]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

if os.path.exists(ZIP):
    with zipfile.ZipFile(ZIP, "r") as zf:
        zf.extractall(ROOT)
    print(f"已解压：{ZIP} -> {ROOT}")
else:
    print(f"未找到 {ZIP}，请确认或将图片放入：{ROOT}")

EXTS = (".png",".jpg",".jpeg",".bmp",".tif",".tiff")
def list_images(root):
    res = []
    for r,_,fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(EXTS):
                res.append(os.path.join(r,f))
    return sorted(res)

def preprocess(img):
    if img.ndim==3:
        g = rgb2gray(img)
    else:
        g = img.astype(float)
        g = g/255.0 if g.max()>1 else g
    g = rescale_intensity(g, out_range=(0,1))
    g = equalize_adapthist(g, clip_limit=0.01)
    return g

def segment_crack(gray):
    thr = threshold_otsu(gray)
    mask = gray < thr
    mask = binary_opening(mask, square(3))
    mask = binary_closing(mask, square(3))
    mask = remove_small_objects(mask, 80)
    return mask

def endpoints_of_skeleton(skel):
    H,W = skel.shape
    ys, xs = np.nonzero(skel)
    pts = set(zip(xs,ys))
    ends = []
    for (x,y) in pts:
        cnt = 0
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                if dx==0 and dy==0: continue
                nx,ny = x+dx, y+dy
                if 0 <= nx < W and 0 <= ny < H and ((nx,ny) in pts):
                    cnt += 1
        if cnt == 1:
            ends.append((x,y))
    return ends

def order_path_from_skeleton(comp_skel):
    H,W = comp_skel.shape
    ys,xs = np.nonzero(comp_skel)
    if len(xs) < 2:
        return None
    pts = set(zip(xs,ys))
    ends = endpoints_of_skeleton(comp_skel)
    start = ends[0] if len(ends)>0 else (xs[0], ys[0])
    path = [start]; visited = {start}; cur = start
    while True:
        x,y = cur
        nbrs = []
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                if dx==0 and dy==0: continue
                nx,ny = x+dx, y+dy
                if 0 <= nx < W and 0 <= ny < H:
                    if comp_skel[ny, nx] and (nx,ny) not in visited:
                        nbrs.append((nx,ny))
        if not nbrs: break
        if len(path)>=2:
            px,py = path[-2]
            vx,vy = x-px, y-py
            def score(p):
                dx,dy = p[0]-x, p[1]-y
                return -(vx*dx+vy*dy)
            nbrs.sort(key=score)
        nxt = nbrs[0]
        path.append(nxt); visited.add(nxt); cur=nxt
    xs_ord = np.array([p[0] for p in path], dtype=float)
    ys_ord = np.array([p[1] for p in path], dtype=float)
    return xs_ord, ys_ord

def pca_rotate(x, y):
    P = np.vstack([x, y]).T
    mu = P.mean(axis=0)
    Q  = P - mu
    C  = np.cov(Q.T)
    w, V = np.linalg.eigh(C)
    e1 = V[:,1]; e2 = V[:,0]
    E  = np.vstack([e1, e2]).T
    UV = Q @ E
    return UV[:,0], UV[:,1], E, mu

def resample_equal_u(u, v, N):
    idx = np.argsort(u); u=u[idx]; v=v[idx]
    u0,u1 = u[0], u[-1]
    if u1-u0 < 1e-6:
        s = np.r_[0, np.cumsum(np.hypot(np.diff(u), np.diff(v)))]
        ss = np.linspace(0, s[-1], N+1)
        uu = np.interp(ss, s, u); vv = np.interp(ss, s, v)
        return uu, vv
    us = np.linspace(u0, u1, N+1)
    vs = np.interp(us, u, v)
    return us, vs

def resample_equal_arclength(u, v, N):
    s  = np.r_[0, np.cumsum(np.hypot(np.diff(u), np.diff(v)))]
    if s[-1] < 1e-9: return u.copy(), v.copy()
    ss = np.linspace(0, s[-1], N+1)
    us = np.interp(ss, s, u); vs = np.interp(ss, s, v)
    return us, vs

def curvature(u, v):
    du  = np.gradient(u); dv  = np.gradient(v)
    d2u = np.gradient(du); d2v = np.gradient(dv)
    return np.abs(du*d2v - dv*d2u) / ((du*du + dv*dv)**1.5 + 1e-9)

def resample_adaptive_curvature(u, v, N):
    s = np.r_[0, np.cumsum(np.hypot(np.diff(u), np.diff(v)))]
    if s[-1] < 1e-9: return u.copy(), v.copy()
    M  = max(400, N*4)
    ss = np.linspace(0, s[-1], M)
    us = np.interp(ss, s, u); vs = np.interp(ss, s, v)
    k  = curvature(us, vs)
    w  = np.sqrt(k*k + 1e-6)
    cw = np.cumsum(w); cw = (cw - cw.min()) / (cw.max()-cw.min() + 1e-12)
    targets = np.linspace(0, 1, N+1)
    idx = np.searchsorted(cw, targets, side="left"); idx = np.clip(idx, 0, M-1)
    return us[idx], vs[idx]

def z2_from_profile(u, v):
    du = np.diff(u); dv = np.diff(v)
    du = np.where(np.abs(du)<1e-9, np.sign(du)*1e-9 + (du==0)*1e-9, du)
    slope2 = (dv/du)**2
    return float(np.sqrt(np.mean(slope2)))

def JRC_from_z2(z2):
    return 51.85*(z2**0.6) - 10.37

def fit_sine_on_uv(u, v):
    if len(u) < 20: return None
    u0, u1 = u.min(), u.max()
    P0 = max(u1 - u0, 10.0)
    R0 = (np.percentile(v,95)-np.percentile(v,5))/2.0
    C0 = np.median(v); b0 = 0.0
    def ffun(u, R, P, b, C):
        return R*np.sin(2*np.pi*u/P + b) + C
    try:
        popt,_ = curve_fit(ffun, u, v, p0=[R0,P0,b0,C0], maxfev=20000)
        R,P,b,C = popt
        yhat = ffun(u, *popt)
        r2   = 1 - np.sum((v-yhat)**2)/(np.sum((v-v.mean())**2) + 1e-9)
        return dict(R=R, P=P, beta=b, C=C, r2=float(r2))
    except Exception:
        return None

def connected_area_from_mask(mask, skel_comp):
    lbl = label(mask)
    if lbl.max()==0: return 0
    counts = {}
    comp = dilation(skel_comp, square(3))
    ys, xs = np.nonzero(comp)
    for y,x in zip(ys, xs):
        lab = lbl[y,x]
        if lab>0: counts[lab] = counts.get(lab,0)+1
    if len(counts)==0:
        areas = [p.area for p in regionprops(lbl)]
        return max(areas) if len(areas) else 0
    best_lab = max(counts, key=counts.get)
    return int((lbl==best_lab).sum())

# ===================== 主流程 =====================
imgs = list_images(ROOT)
rows = []
N_list = [50,100,200]
focus_names = {"图3-1","图3-2","图3-3"}

for path in imgs:
    img  = skio.imread(path)
    gray = preprocess(img)
    mask = segment_crack(gray)
    skel = skeletonize(mask)
    lbl_s  = label(skel)
    propsS = regionprops(lbl_s)

    for ridx, r in enumerate(propsS):
        comp_skel = (lbl_s == r.label)
        ordered   = order_path_from_skeleton(comp_skel)
        if ordered is None:
            continue
        xs, ys = ordered
        u, v, E, mu = pca_rotate(xs, ys)
        area_px = connected_area_from_mask(mask, comp_skel)
        result_by_N = {}
        for N in N_list:
            u_eu, v_eu = resample_equal_u(u, v, N)
            u_es, v_es = resample_equal_arclength(u, v, N)
            u_ca, v_ca = resample_adaptive_curvature(u, v, N)
            z2_eu = z2_from_profile(u_eu, v_eu)
            z2_es = z2_from_profile(u_es, v_es)
            z2_ca = z2_from_profile(u_ca, v_ca)
            result_by_N[N] = dict(
                z2_equal_u=z2_eu,        JRC_equal_u=JRC_from_z2(z2_eu),
                z2_equal_s=z2_es,        JRC_equal_s=JRC_from_z2(z2_es),
                z2_curv_adapt=z2_ca,     JRC_curv_adapt=JRC_from_z2(z2_ca)
            )
        sine_fit = fit_sine_on_uv(u, v)
        row = dict(
            image=os.path.basename(path), frac_id=int(ridx+1), area_px=int(area_px),
            JRC_equal_u_200=result_by_N[200]["JRC_equal_u"],
            JRC_equal_s_200=result_by_N[200]["JRC_equal_s"],
            JRC_curv_adapt_200=result_by_N[200]["JRC_curv_adapt"],
            z2_equal_u_200=result_by_N[200]["z2_equal_u"],
            z2_equal_s_200=result_by_N[200]["z2_equal_s"],
            z2_curv_adapt_200=result_by_N[200]["z2_curv_adapt"],
            JRC_equal_u_50=result_by_N[50]["JRC_equal_u"],
            JRC_equal_u_100=result_by_N[100]["JRC_equal_u"],
        )
        if sine_fit is not None:
            row.update(dict(R_px=float(sine_fit["R"]), P_px=float(sine_fit["P"]),
                            beta=float(sine_fit["beta"]), C_px=float(sine_fit["C"]),
                            sine_r2=float(sine_fit["r2"])))
        rows.append(row)

        basename = os.path.splitext(os.path.basename(path))[0]
        if (basename in focus_names) or (ridx < 2):
            fig, axes = plt.subplots(1, 4, figsize=(14, 8))
            axes[0].imshow(gray, cmap="gray"); axes[0].set_title("原图"); axes[0].axis("off")
            axes[1].imshow(mask, cmap="gray"); axes[1].set_title("裂隙掩膜"); axes[1].axis("off")
            axes[2].imshow(gray, cmap="gray")
            ys_plot, xs_plot = np.nonzero(comp_skel)
            axes[2].plot(xs_plot, ys_plot, '.', markersize=0.6)
            axes[2].set_title("骨架"); axes[2].axis("off")
            axes[3].imshow(gray, cmap="gray")
            us, vs = resample_equal_u(u, v, 200)
            P_samp = np.vstack([us, vs]).T @ np.array(E).T + mu
            axes[3].plot(P_samp[:, 0], P_samp[:, 1], linewidth=1.5)
            axes[3].set_title("等u采样叠加(N=200)"); axes[3].axis("off")
            fig.tight_layout()
            p_out = os.path.join(OUT, f"{basename}_frac{ridx + 1}_panel.png")
            fig.savefig(p_out, bbox_inches="tight", dpi=200); plt.close(fig)

# ===================== 导出 CSV =====================
df = pd.DataFrame(rows)
csv_main = os.path.join(OUT, "问题3_JRC与拟合结果.csv")
df.to_csv(csv_main, index=False, encoding='utf-8-sig')
print(f"已保存 CSV 文件：{csv_main}")

# 生成论文用表（表2）
table_cols = ["图像编号","裂隙编号","振幅R(px)","周期P(px)","相位β(rad)","中心线位置C(px)","JRC值(等u,N=200)"]
rows_table = []
for _,r in df.iterrows():
    Rpx=Ppx=Cpx=""
    if 'R_px' in r and not pd.isna(r['R_px']):
        Rpx = float(r['R_px']); Ppx = float(r['P_px']); Cpx = float(r['C_px'])
    rows_table.append([r['image'], int(r['frac_id']), Rpx, Ppx, r.get('beta',''), Cpx, float(r['JRC_equal_u_200'])])
df_table = pd.DataFrame(rows_table, columns=table_cols)
table_csv = os.path.join(OUT, '问题3_表2数据.csv')
df_table.to_csv(table_csv, index=False, encoding='utf-8-sig')
print(f"已保存 论文表2 CSV：{table_csv}")

# ===================== 汇总可视化=====================
if len(df):
    x = df['area_px'].values.astype(float)
    y = df['JRC_equal_u_200'].values.astype(float)
    fig = plt.figure(figsize=(10,6))
    plt.bar(range(len(x)), y, color=plt.cm.tab20.colors[:len(x)])
    plt.xticks(range(len(x)), [f"裂隙{idx+1}" for idx in range(len(x))], rotation=45)
    plt.xlabel("裂隙编号"); plt.ylabel("JRC (等u, N=200)")
    plt.title("不同裂隙面积对应的JRC值柱状图")
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    out1 = os.path.join(OUT, '问题3_JRC_vs_area_柱状图.png')
    fig.tight_layout()
    fig.savefig(out1, bbox_inches='tight', dpi=200); plt.close(fig)
    print(f"已保存：{out1}")

    avg_JRCs = [df[f'JRC_equal_u_{N}'].mean() for N in [50,100,200]]
    labels = ['N=50','N=100','N=200']
    fig = plt.figure(figsize=(6,6))
    plt.pie(avg_JRCs, labels=labels, autopct='%1.1f%%', colors=plt.cm.Set2.colors)
    plt.title("不同采样密度下JRC平均值占比饼图")
    out2 = os.path.join(OUT, '问题3_采样敏感性饼图.png')
    fig.savefig(out2, bbox_inches='tight', dpi=200); plt.close(fig)
    print(f"已保存：{out2}")

print("问题3分析完成 ✅")
