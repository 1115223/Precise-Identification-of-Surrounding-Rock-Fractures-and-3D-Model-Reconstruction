# Precise-Identification-of-Surrounding-Rock-Fractures-and-3D-Model-Reconstruction
As the primary source of my country's energy supply, the safe and efficient extraction of coal relies heavily on the precise identification of fractures in the surrounding rock of mine roadways.
### 安装依赖
```bash
pip install -r requirements.txt
```


## 系统功能

### 核心功能
- ✅ 自动识别钻孔成像展开图中的裂隙像素
- ✅ 生成二值化结果（裂隙为黑色，其他为白色）
- ✅ 排除干扰因素（岩石纹理、钻头痕迹、泥浆污染等）
- ✅ 提供调试可视化图像

### 输出文件
- `binary_*.jpg`: 二值化结果图像
- `debug_*.jpg`: 调试可视化图像（包含各处理步骤）

## 裂隙识别关键参数
self.min_fracture_width = 1.0  # 最小裂隙宽度(mm)
self.pixel_to_mm_ratio = 0.1   # 像素到毫米的转换比例
self.gaussian_kernel_size = 5   # 高斯滤波核大小
self.edge_threshold_low = 50    # Canny边缘检测低阈值
self.edge_threshold_high = 150  # Canny边缘检测高阈值
```

### 调整建议
1. **像素转换比例**: 根据实际图像调整 `pixel_to_mm_ratio`
2. **边缘检测阈值**: 根据图像对比度调整 `edge_threshold_low/high`
3. **最小裂隙宽度**: 根据实际需求调整 `min_fracture_width`

## 故障排除

### 常见问题
1. **OpenCV错误**: 确保安装了正确版本的OpenCV
2. **内存不足**: 处理大图像时可能需要更多内存
3. **检测效果不佳**: 调整参数或检查图像质量

### 调试方法
1. 查看 `debug_*.jpg` 文件了解各处理步骤
2. 调整参数重新运行
3. 检查输入图像质量

## 技术特点

### 算法优势
- 多算法融合的边缘检测
- 基于几何特征的裂隙判断
- 形态学操作优化结果
- 有效抑制干扰因素

```

## 联系支持
如有问题，请检查：
1. 依赖包是否正确安装
2. 输入图像路径是否正确
3. 参数设置是否合适
