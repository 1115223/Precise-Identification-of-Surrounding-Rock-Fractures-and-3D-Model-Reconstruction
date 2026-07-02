# q4.py 详细解释文档

## 概述
q4.py是一个用于分析钻孔岩芯图像中裂隙特征的完整脚本，主要功能包括：
- 自动解压附件数据
- 图像预处理与裂隙分割
- 裂隙几何特征提取
- 平面产状计算
- 彩色可视化输出
- 结果统计分析

## 主要依赖库及作用

### 基础库
- `os, re, zipfile, shutil`: 文件系统操作、正则表达式、压缩包处理
- `math, json, warnings`: 数学计算、JSON处理、警告管理
- `numpy (np)`: 数值计算和数组操作
- `pandas (pd)`: 数据框操作和CSV处理
- `matplotlib.pyplot (plt)`: 图像显示和保存

### 图像处理库 (scikit-image)
- `skimage.io`: 图像读取
- `skimage.color.rgb2gray`: 彩色转灰度
- `skimage.filters`: 阈值分割、边缘检测
- `skimage.exposure`: 图像对比度增强
- `skimage.morphology`: 形态学操作
- `skimage.measure`: 区域分析、轮廓提取

### 可视化库
- `seaborn (sns)`: 美化图表样式

## 核心算法和方法详解

### 1. 图像预处理 (`gray_preprocess`)
**位置**: q4.py:69-75
```python
def gray_preprocess(img):
    if img.ndim==3: g = rgb2gray(img)     # 彩色转灰度
    else: g = img.astype(float)           # 已是灰度图
    if g.max() > 1: g = g/255.0           # 归一化到[0,1]
    g = rescale_intensity(g, out_range=(0,1))  # 拉伸对比度
    g = equalize_adapthist(g, clip_limit=0.01) # 自适应直方图均衡化
    return g
```
**方法说明**:
- **rgb2gray**: 将RGB图像转换为灰度图像，使用标准加权公式
- **rescale_intensity**: 线性拉伸图像强度到指定范围，增强对比度
- **equalize_adapthist**: 自适应局部直方图均衡化，增强细节对比度

### 2. 裂隙分割 (`crack_segment`)
**位置**: q4.py:77-85
```python
def crack_segment(gray_img):
    edge_map = sobel(gray_img)            # Sobel边缘检测
    thresh = threshold_otsu(gray_img)     # Otsu自动阈值
    mask_intensity = gray_img < thresh    # 强度阈值掩码
    mask = (edge_map > np.percentile(edge_map, 70)) | mask_intensity  # 综合掩码
    mask = binary_opening(mask, square(3))   # 形态学开运算
    mask = binary_closing(mask, square(3))   # 形态学闭运算
    mask = remove_small_objects(mask, 120)   # 移除小连通域
    return mask
```
**方法说明**:
- **sobel**: Sobel算子进行边缘检测，突出裂隙边界
- **threshold_otsu**: Otsu方法自动计算最优二值化阈值
- **binary_opening**: 先腐蚀后膨胀，消除噪声点
- **binary_closing**: 先膨胀后腐蚀，连接断裂区域
- **remove_small_objects**: 移除面积小于阈值的连通域

### 3. 主轮廓提取 (`extract_main_contour`)
**位置**: q4.py:87-98
```python
def extract_main_contour(bin_mask):
    lbl_img = label(bin_mask)             # 连通域标记
    props = regionprops(lbl_img)          # 区域属性分析
    props.sort(key=lambda r: r.area, reverse=True)  # 按面积排序
    comp = (lbl_img == props[0].label).astype(float) # 最大连通域
    contours = find_contours(comp, 0.5)   # 轮廓提取
    contours.sort(key=lambda c: len(c), reverse=True)  # 按长度排序
    c = contours[0]                       # 最长轮廓
    return c[:,1], c[:,0]                 # 返回x,y坐标
```
**方法说明**:
- **label**: 对二值图像进行连通域标记，每个连通区域分配唯一标签
- **regionprops**: 计算各连通域的几何属性（面积、重心等）
- **find_contours**: 提取等值线轮廓，0.5为阈值

### 4. 主成分分析 (`compute_pca`)
**位置**: q4.py:100-109
```python
def compute_pca(u_pts, v_pts):
    pts = np.vstack([u_pts, v_pts]).T     # 构建点集矩阵
    mu = pts.mean(axis=0)                 # 计算中心点
    q = pts - mu                          # 中心化
    cov = np.cov(q.T)                     # 协方差矩阵
    w, V = np.linalg.eigh(cov)            # 特征值分解
    e1 = V[:,1]; e2 = V[:,0]              # 主成分方向
    E = np.vstack([e1,e2]).T              # 变换矩阵
    uv = q @ E                            # 坐标变换
    return uv[:,0], uv[:,1], E, mu
```
**方法说明**:
- **协方差矩阵**: 描述数据各维度间的线性相关性
- **特征值分解**: 找到数据的主要变化方向
- **坐标变换**: 将原坐标系转换到主成分坐标系

### 5. 等间距采样 (`equal_sample`)
**位置**: q4.py:111-120
```python
def equal_sample(u,v,N=200):
    idx = np.argsort(u); u=u[idx]; v=v[idx]  # 按u排序
    if (u[-1]-u[0]) < 1e-6:               # u范围过小
        s = np.r_[0, np.cumsum(np.hypot(np.diff(u), np.diff(v)))]  # 弧长参数
        ss = np.linspace(0, s[-1], N+1)   # 等弧长采样
        uu = np.interp(ss, s, u); vv = np.interp(ss, s, v)
        return uu, vv
    us = np.linspace(u[0], u[-1], N+1)    # 等u间距采样
    vs = np.interp(us, u, v)              # 插值计算v
    return us, vs
```
**方法说明**:
- **弧长参数化**: 当u变化范围很小时，使用弧长作为参数
- **线性插值**: 在已知点间进行线性插值，获得均匀采样点

### 6. 粗糙度计算 (`compute_z2`, `z2_to_jrc`)
**位置**: q4.py:122-128
```python
def compute_z2(u, v):
    du = np.diff(u); dv = np.diff(v)      # 计算差分
    du = np.where(np.abs(du)<1e-9, np.sign(du)*1e-9 + (du==0)*1e-9, du)  # 避免除零
    return float(np.sqrt(np.mean((dv/du)**2)))  # Z2粗糙度系数

def z2_to_jrc(z2_val):
    return 51.85*(z2_val**0.6) - 10.37    # 经验公式转换JRC
```
**方法说明**:
- **Z2系数**: 反映裂隙表面的起伏程度，定义为坡度变化的均方根
- **JRC转换**: 使用Barton经验公式将Z2转换为节理粗糙度系数

### 7. 正弦函数拟合 (`sine_fit`)
**位置**: q4.py:130-142
```python
def sine_fit(u, v):
    P = max(u1-u0, 10.0)                  # 周期估计
    w = 2*np.pi/P                         # 角频率
    S = np.sin(w*u); C = np.cos(w*u)      # 构建基函数
    A = np.vstack([S, C, np.ones_like(u)]).T  # 设计矩阵
    sol, *_ = np.linalg.lstsq(A, v, rcond=None)  # 最小二乘拟合
    a,b,c = sol                           # 拟合系数
    R = float(np.hypot(a,b))              # 振幅
    beta = float(np.arctan2(b, a))        # 相位角
    yhat = a*S + b*C + c                  # 拟合值
    r2_val = 1 - np.sum((v-yhat)**2)/(np.sum((v-v.mean())**2)+1e-12)  # R²
    return dict(R=R, P=P, beta=beta, C=float(c), r2=r2_val)
```
**方法说明**:
- **最小二乘法**: 通过最小化残差平方和来拟合正弦函数
- **R²系数**: 衡量拟合优度，值越接近1表示拟合越好

### 8. 平面产状计算 (`plane_from_sine`)
**位置**: q4.py:144-155
```python
def plane_from_sine(sf, img_shape, depth, radius_mm):
    H, W = img_shape
    mm_per_px = depth / float(H)          # 像素毫米比例
    R_mm = sf["R"] * mm_per_px            # 振幅转换为毫米
    dip = math.atan2(max(R_mm,0.0), radius_mm)  # 倾角计算
    az = sf["beta"]                       # 方位角
    C_mm = sf["C"] * mm_per_px            # 偏移量
    hz = 1.0 / max(math.tan(dip), 1e-6)   # 法向量z分量
    nx = math.cos(az); ny = math.sin(az)  # 法向量x,y分量
    n = np.array([nx, ny, hz], dtype=float)
    n = n / np.linalg.norm(n)             # 单位化
    return dict(R_mm=R_mm, dip=dip, az=az, C_mm=C_mm, normal=n)
```
**方法说明**:
- **倾角计算**: 通过arctangent函数计算裂隙面与水平面的夹角
- **法向量**: 根据方位角和倾角计算裂隙面的单位法向量

### 9. 钻孔识别 (`detect_hole`)
**位置**: q4.py:157-167
```python
def detect_hole(fname):
    base = os.path.basename(fname)
    m = re.search(r'([1-6])#', base)      # 查找"数字#"模式
    if m: return f"{m.group(1)}#"
    m = re.search(r'(?:BH|孔|Hole|H)([1-6])', base, re.IGNORECASE)  # 其他模式
    if m: return f"{m.group(1)}#"
    # 在路径中查找
    parts = os.path.normpath(fname).split(os.sep)
    for p in parts:
        m = re.search(r'([1-6])#', p)
        if m: return f"{m.group(1)}#"
    return None
```
**方法说明**:
- **正则表达式**: 使用多种模式匹配文件名中的钻孔编号
- **路径解析**: 在完整路径的各级目录中查找钻孔标识

### 10. 彩色叠加可视化 (`overlay_crack`)
**位置**: q4.py:174-180
```python
def overlay_crack(img, mask):
    if img.ndim==2: img_color = np.stack([img]*3, axis=-1)  # 灰度转RGB
    else: img_color = img.copy()
    img_color = img_color.astype(float)/img_color.max()  # 归一化
    img_color[mask>0] = [1,0,0]           # 裂隙区域标红
    return img_color
```
**方法说明**:
- **颜色空间转换**: 确保输出为RGB格式
- **掩码叠加**: 将检测到的裂隙区域标记为红色

## 数据结构

### 钻孔信息 (q4.py:46-53)
```python
borehole_info = pd.DataFrame([
    {"id": "1#", "x": 500, "y": 2000, "depth": 7000},
    # ... 其他钻孔
])
```
包含每个钻孔的ID、坐标位置和深度信息。

### 输出记录结构
每个处理的图像生成一条记录，包含：
- 基本信息：图像名、钻孔号、尺寸
- 几何参数：深度、Z2系数、JRC值
- 拟合参数：振幅R、周期P、相位beta、偏移C、拟合度r2
- 产状参数：倾角dip、方位角az、法向量normal
- 位置信息：钻孔坐标、平面参考点

## 输出文件

1. **问题4结果/输出/问题4_裂隙平面.csv**: 所有裂隙分析结果的CSV表格
2. **问题4结果/图件彩色/*.png**: 彩色裂隙叠加图像
3. **问题4结果/输出/问题4_摘要.json**: 处理统计摘要
4. **问题4结果/图件彩色/问题4_钻孔饼图.png**: 钻孔裂隙分布饼图

## 算法流程

1. **数据准备**: 解压附件，扫描图像文件
2. **图像预处理**: 灰度转换、对比度增强
3. **裂隙检测**: 边缘检测+阈值分割+形态学处理
4. **轮廓提取**: 连通域分析，提取最大裂隙轮廓
5. **几何分析**: PCA主成分分析，等间距重采样
6. **特征计算**: Z2粗糙度、JRC转换、正弦拟合
7. **产状计算**: 倾角、方位角、法向量
8. **结果输出**: CSV表格、可视化图件、统计摘要

## 关键参数设置

- **形态学核大小**: square(3) - 3×3正方形结构元素
- **小对象阈值**: 120像素 - 移除过小的噪声区域
- **边缘阈值**: 70百分位数 - 保留主要边缘特征
- **采样点数**: 200个 - 确保足够的分析精度
- **圆周长**: π×30mm - 基于钻孔直径的理论计算

这个脚本实现了从原始岩芯图像到裂隙几何参数的完整分析流程，结合了多种图像处理和数值分析方法。