本项目实现了从裂隙图像到正弦函数拟合曲线的完整处理流程，主要包括图像处理、特征提取、数学建模和参数优化等步骤。

### 核心目标
- 从二值化图像中自动识别裂隙线条
- 将裂隙轨迹拟合为正弦函数：`y = A * sin(B * x + C) + D`
- 输出精确的数学参数和可视化结果

---

## 图像预处理阶段

### 1.1 图像加载与格式转换

```python
# 核心代码位置：sinusoidal_function_fitter.py:33-52
gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
```

**详细步骤：**
1. **读取图像**：使用OpenCV读取图像文件
2. **灰度转换**：确保图像为单通道灰度图
3. **二值化检查**：确保裂隙为白色(255)，背景为黑色(0)
4. **图像反转**：如果需要，自动反转图像确保正确的对比度

**处理原理：**
- 二值化图像便于后续的形态学操作
- 统一的颜色编码（白色裂隙，黑色背景）确保算法的一致性

### 1.2 坐标系转换

```python
# 核心代码位置：sinusoidal_function_fitter.py:54-66
def convert_to_mathematical_coordinates(self, points, image_height):
    math_points = points.copy().astype(np.float64)
    math_points[:, 1] = image_height - math_points[:, 1]
    return math_points
```

**转换说明：**
- **图像坐标系**：原点(0,0)在左上角，y轴向下
- **数学坐标系**：原点(0,0)在左下角，y轴向上
- **转换公式**：
  ```
  x_math = x_image
  y_math = image_height - y_image
  ```

**重要性：**
这一步确保拟合的正弦函数符合数学惯例，便于后续的数学分析和参数解释。

---

## 裂隙特征提取

### 2.1 轮廓检测

```python
# 核心代码位置：sinusoidal_function_fitter.py:68-113
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

**检测流程：**
1. **轮廓提取**：使用OpenCV的`findContours`函数
2. **外部轮廓**：`cv2.RETR_EXTERNAL`只检测最外层轮廓
3. **简化近似**：`cv2.CHAIN_APPROX_SIMPLE`压缩水平、垂直和对角线段

### 2.2 轮廓筛选与分析

```python
# 筛选条件
if arc_length < self.min_contour_length:  # 最小弧长：20像素
    continue
if x_range < self.min_x_range:           # 最小x范围：30像素
    continue
if y_range < self.min_y_range:           # 最小y范围：3像素
    continue
```

**筛选标准：**
- **弧长阈值**：过滤掉过短的噪声线段
- **范围阈值**：确保裂隙有足够的空间变化
- **点数阈值**：保证有足够的数据点进行拟合

**轮廓属性计算：**
```python
arc_length = cv2.arcLength(contour, False)  # 弧长
area = cv2.contourArea(contour)             # 面积
x_range = np.max(x_coords) - np.min(x_coords)  # x范围
y_range = np.max(y_coords) - np.min(y_coords)  # y范围
```

---

## 数据预处理与函数化

### 3.1 点数据去重与平均

```python
# 核心代码位置：sinusoidal_function_fitter.py:115-133
unique_x, indices, counts = np.unique(x_coords, return_inverse=True, return_counts=True)
unique_y = np.array([np.mean(y_coords[indices == i]) for i in range(len(unique_x))])
```

**处理目的：**
- **函数化**：确保每个x值对应唯一的y值
- **降噪**：通过平均化减少测量噪声
- **数据压缩**：减少冗余数据点

### 3.2 数据平滑处理

```python
# Savitzky-Golay滤波
if len(unique_y) > 5:
    window_size = min(5, len(unique_y) // 3)
    if window_size >= 3:
        smoothed_y = np.convolve(unique_y, np.ones(window_size)/window_size, mode='same')
```

**平滑算法：**
- **移动平均**：简单有效的局部平滑
- **窗口大小**：自适应窗口，避免过度平滑
- **边界处理**：使用`mode='same'`保持数据长度

### 3.3 分段拟合策略

```python
# 核心代码位置：fit_sine_muli.py:30-76
def fit_band(gray, top, bottom, min_pix=600, r2_th=0.35):
    band = inv[top:bottom, :]  # 提取水平带
    # 每列取中位数实现函数化
    df = pd.DataFrame({'x': xs, 'y': ys}).groupby('x')['y'].median()
```

**分段原理：**
- **水平分带**：将图像分为多个水平条带
- **独立拟合**：每个条带独立进行拟合
- **参数继承**：为全局拟合提供初始参数

---

## 正弦函数拟合

### 4.1 数学模型定义

```python
# 正弦函数模型
def sinusoidal_function(self, x, R, P, a, c):
    return R * np.sin(2 * np.pi * x / P + a) + c
```

**参数含义：**
- **R (振幅)**：正弦波的最大偏移量
- **P (周期)**：完成一个完整振荡的x距离
- **a (相位)**：水平偏移量（弧度）
- **c (中心线)**：正弦波的垂直偏移

### 4.2 初始参数估计

#### 4.2.1 统计学方法

```python
# 基本统计量估计
y_mean = np.mean(y_data)      # 中心线
y_std = np.std(y_data)        # 标准差
y_range = np.max(y_data) - np.min(y_data)  # 范围

# 参数估计
R_init = max(y_range / 2, y_std)    # 振幅
c_init = y_mean                      # 中心线
```

#### 4.2.2 频域分析方法

```python
# FFT频率分析
fft = np.fft.fft(y_detrend)
freqs = np.fft.fftfreq(len(y_detrend), d=1)
power = np.abs(fft[1:len(fft)//2])
main_freq_idx = np.argmax(power) + 1
main_freq = abs(freqs[main_freq_idx])

# 周期估计
if main_freq > 0:
    P_init = 1.0 / main_freq * len(x_data) / x_range
```

**FFT分析步骤：**
1. **去趋势**：减去均值消除直流分量
2. **快速傅里叶变换**：转换到频域
3. **功率谱分析**：找到主要频率成分
4. **周期计算**：从频率反推周期

### 4.3 多初值优化策略

```python
# 多组初始参数
initial_guesses = [
    [R_init, P_init, a_init, c_init],           # 基础估计
    [R_init*0.5, P_init*0.8, np.pi/4, c_init], # 小振幅变体
    [R_init*1.5, P_init*1.2, -np.pi/4, c_init],# 大振幅变体
    [R_init, P_init/2, np.pi/2, c_init],        # 短周期变体
    [R_init, P_init*2, -np.pi/2, c_init]       # 长周期变体
]
```

**优化原理：**
- **全局搜索**：多个起点避免局部最优
- **参数变体**：系统性地探索参数空间
- **最优选择**：选择R²最高的拟合结果

### 4.4 约束优化

```python
# 参数边界设置
bounds = (
    [0.1, 10, -4*np.pi, y_mean - y_range*2],           # 下界
    [y_range*3, x_range*3, 4*np.pi, y_mean + y_range*2] # 上界
)

# 非线性拟合
popt, pcov = curve_fit(
    self.sinusoidal_function,
    x_data, y_data,
    p0=initial_guess,
    bounds=bounds,
    maxfev=8000,      # 最大函数评估次数
    ftol=1e-8,        # 函数容差
    xtol=1e-8         # 参数容差
)
```

**约束说明：**
- **振幅约束**：[0.1, y_range*3] 确保振幅为正且合理
- **周期约束**：[10, x_range*3] 避免过短或过长周期
- **相位约束**：[-4π, 4π] 覆盖所有可能相位
- **中心约束**：围绕均值的合理范围

---

## 结果优化与合并

### 5.1 相似曲线聚类

```python
# 核心代码位置：fit_sine_muli.py:92-103
def curves_close(f1, f2):
    A1,B1,C1,D1 = f1["A"], f1["B"], f1["C"], f1["D"]
    A2,B2,C2,D2 = f2["A"], f2["B"], f2["C"], f2["D"]

    # 参数相似性检查
    if abs(D1-D2) > d_tol: return False      # 中心线距离
    if abs(B1-B2) > b_tol: return False      # 频率差异
    if abs(A1-A2)/max(A1,A2) > a_rel_tol: return False  # 振幅相对差异

    # 曲线形状相似性
    mean_abs_diff = np.mean(np.abs(y1 - y2))
    return mean_abs_diff <= curve_eps_ratio * max(A1, A2)
```

**聚类标准：**
- **中心线容差**：d_tol = 80像素
- **频率容差**：b_tol = 0.025
- **振幅相对容差**：a_rel_tol = 35%
- **形状相似度**：curve_eps_ratio = 18%

### 5.2 全局重拟合

```python
# 合并数据点进行全局拟合
xs = np.concatenate([fits[k]["x_samples"] for k in group])
ys = np.concatenate([fits[k]["y_samples"] for k in group])
order = np.argsort(xs)
xs, ys = xs[order], ys[order]

# 使用最佳分段结果作为初值
best = max(group, key=lambda k: fits[k]["R2"])
p0 = [fits[best]["A"], fits[best]["B"], fits[best]["C"], fits[best]["D"]]
```

**重拟合优势：**
- **数据完整性**：使用所有相关数据点
- **参数一致性**：消除分段拟合的不连续性
- **质量提升**：通常获得更好的拟合质量

### 5.3 覆盖率验证

```python
# 检查拟合曲线与原始裂隙的重合度
xcheck = np.linspace(0, W0-1, 1000)
ycheck = sine_func(xcheck, *popt)
ycheck = np.clip(np.round(ycheck).astype(int), 0, H0-1)

# 计算覆盖率
cnt_hit = 0
for xx, yy in zip(xcheck.astype(int), ycheck):
    if mask[yy, xx]:  # mask是原始裂隙像素
        cnt_hit += 1
coverage = cnt_hit / cnt_total

# 阈值筛选
if coverage < coverage_thresh:  # 默认50%
    keep_flag = False
```

**验证目的：**
- **真实性检查**：确保拟合曲线真实反映裂隙位置
- **噪声排除**：过滤掉偏离实际裂隙的拟合结果
- **质量保证**：提高最终结果的可靠性

---

## 质量评估

### 6.1 统计指标

#### 6.1.1 决定系数 (R²)

```python
ss_res = np.sum((y_data - y_fitted)**2)    # 残差平方和
ss_tot = np.sum((y_data - y_mean)**2)      # 总平方和
r_squared = 1 - (ss_res / ss_tot)
```

**R²解释：**
- **范围**：[0, 1]，越接近1表示拟合越好
- **含义**：解释变量能够解释的方差比例
- **阈值**：通常要求R² > 0.35才认为拟合有效

#### 6.1.2 均方根误差 (RMSE)

```python
rmse = np.sqrt(np.mean((y_data - y_fitted)**2))
```

**RMSE特点：**
- **单位**：与原始数据相同单位（像素）
- **敏感性**：对大误差更敏感
- **直观性**：直接反映拟合的绝对精度

#### 6.1.3 平均绝对误差 (MAE)

```python
mae = np.mean(np.abs(y_data - y_fitted))
```

**MAE优势：**
- **鲁棒性**：对异常值不敏感
- **直观性**：平均偏离距离
- **稳定性**：不受极端值影响

### 6.2 参数不确定性

```python
param_errors = np.sqrt(np.diag(pcov))  # 参数标准误差
```

**不确定性来源：**
- **测量噪声**：图像数字化和像素化误差
- **模型误差**：真实裂隙可能不完全符合正弦模型
- **数值误差**：优化算法的收敛精度

---

## 可视化输出

### 7.1 多层次可视化

#### 7.1.1 原始图像叠加

```python
# 在原始图像上绘制拟合曲线
fig = plt.figure(figsize=(4, 10))
ax = plt.gca()
ax.imshow(gray, cmap='gray')
for r in merged:
    xfit = np.linspace(0, W-1, 1500)
    yfit = sine_func(xfit, r["A"], r["B"], r["C"], r["D"])
    ax.plot(xfit, yfit, '-', lw=2)
```

#### 7.1.2 数学坐标系显示

```python
# 数学坐标系下的拟合结果
axes[0, 1].scatter(x_data, y_data, alpha=0.6, s=20, label='数据点')
axes[0, 1].plot(x_data, y_fitted, linewidth=2, label='拟合曲线')
axes[0, 1].set_xlabel('x坐标')
axes[0, 1].set_ylabel('y坐标')
```

#### 7.1.3 质量对比图

```python
# 拟合质量对比
r_squared_values = [r['r_squared'] for r in fitted_results]
rmse_values = [r['rmse'] for r in fitted_results]

bars1 = axes[1, 0].bar(x_pos, r_squared_values, label='R²')
bars2 = ax3_twin.bar(x_pos, rmse_values, label='RMSE')
```

### 7.2 参数汇总表

```python
# 创建参数表格
headers = ['裂隙#', 'R(振幅)', 'P(周期)', 'a(相位)', 'c(中心)', 'R²']
table_data = []
for i, result in enumerate(fitted_results):
    row = [f"#{i+1}", f"{result['R']:.3f}", f"{result['P']:.3f}",
           f"{result['a']:.3f}", f"{result['c']:.3f}", f"{result['r_squared']:.3f}"]
    table_data.append(row)
```

### 7.3 详细报告生成

```python
# 生成文本报告
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("正弦函数拟合分析报告\n")
    f.write("=" * 60 + "\n\n")

    for i, result in enumerate(fitted_results):
        f.write(f"裂隙线 #{i+1}\n")
        f.write(f"  振幅 R = {result['R']:.6f} ± {result['param_errors'][0]:.6f}\n")
        f.write(f"  周期 P = {result['P']:.6f} ± {result['param_errors'][1]:.6f}\n")
        f.write(f"  相位 a = {result['a']:.6f} ± {result['param_errors'][2]:.6f}\n")
        f.write(f"  中心 c = {result['c']:.6f} ± {result['param_errors'][3]:.6f}\n")
        f.write(f"  R² = {result['r_squared']:.6f}\n")
        f.write(f"  函数: {result['function_equation']}\n\n")
```

---

