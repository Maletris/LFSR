# 多视图光场超分辨率使用说明

## 功能特性

本实现支持对光场图像的**多个视图**同时进行超分辨率重建，相比原始单视图方法具有以下特点：

- ✅ 支持不同输入视图数量：9×9、13×13、15×15
- ✅ 支持不同目标超分区域：5×5、7×7、9×9
- ✅ 自动循环处理所有目标视图
- ✅ 结果保存为 4D 矩阵格式 `[U, V, H, W]`
- ✅ 提供每个视图的 PSNR 矩阵
- ✅ 完全复用原始训练逻辑，无需重写

## 文件说明

### 新增文件

1. **`Main_LFZSSR_MultiView.py`**
   - 多视图训练的主入口脚本
   - 支持命令行参数配置

2. **`trainers/Class_LFZSSR.py`**（已修改）
   - 新增 `LFZSSR_MultiTargetView` 类
   - 包装 `LFZSSR_SingleTargetView` 实现多视图训练

3. **`configs/Config_for_LFZSSR_MultiView.py`**
   - 多视图训练配置文件
   - 继承自原始 `Config` 类

## 使用方法

### 基本用法

```bash
python Main_LFZSSR_MultiView.py --dataset EPFL --scale 2 \
    --input-views 15 --target-views 5 --start 0 --end 1
```

### 命令行参数

#### 基础参数
- `--gpus`: GPU ID（默认: "0"）
- `--scale`: 超分倍率（2 或 3，默认: 2）
- `--dataset`: 数据集名称（Stanford, EPFL, HCI1, HCI2）
- `--start`: 数据集起始索引（默认: 0）
- `--end`: 数据集结束索引（默认: 1）
- `--gpu-num`: GPU 数量（默认: 1）
- `--record`: 是否记录训练日志（添加此参数启用）

#### 多视图专用参数
- `--input-views`: 输入光场视图数量（9/13/15，默认: 9）
- `--target-views`: 目标超分区域大小（5/7/9，默认: 5）

### 使用示例

#### 示例 1：输入 15×15，超分中心 5×5
```bash
python Main_LFZSSR_MultiView.py --dataset EPFL --scale 2 \
    --input-views 15 --target-views 5 --start 0 --end 1
```

#### 示例 2：输入 13×13，超分中心 5×5
```bash
python Main_LFZSSR_MultiView.py --dataset HCI1 --scale 2 \
    --input-views 13 --target-views 5 --start 0 --end 5
```

#### 示例 3：输入 9×9，超分中心 7×7
```bash
python Main_LFZSSR_MultiView.py --dataset Stanford --scale 3 \
    --input-views 9 --target-views 7 --start 0 --end 10
```

#### 示例 4：启用训练日志记录
```bash
python Main_LFZSSR_MultiView.py --dataset EPFL --scale 2 \
    --input-views 15 --target-views 5 --start 0 --end 1 --record
```

## 输出结果

### 结果文件结构

运行后会在 `./results/` 目录下生成以下文件：

```
results/
└── LFZSSR_MultiView_Set_EPFL_Scale_2_Input15x15_Target5x5/
    ├── Bikes/                           # 第一个光场文件夹
    │   ├── models/                      # 该光场的模型文件目录
    │   │   ├── Bikes_view_2_2_0_model_final.pth
    │   │   ├── Bikes_view_2_3_0_model_final.pth
    │   │   ├── Bikes_view_2_4_0_model_final.pth
    │   │   └── ...                      # 共 25 个模型（5x5 视图）
    │   └── multiview_result_Bikes.mat   # 该光场的超分结果
    ├── Danger_de_mort/                  # 第二个光场文件夹
    │   ├── models/
    │   │   ├── Danger_de_mort_view_2_2_0_model_final.pth
    │   │   └── ...
    │   └── multiview_result_Danger_de_mort.mat
    └── summary_all.mat                  # 所有光场的统计汇总（在数据集根目录）
```

**说明：**
- 每个光场（如 Bikes、Danger_de_mort）有独立的文件夹
- 每个光场文件夹内有：
  - `models/` 子文件夹：存放该光场所有视图的训练模型（.pth 文件）
  - `multiview_result_<lf_name>.mat`：该光场的超分结果和 PSNR
- `summary_all.mat`：所有光场的统计汇总（位于数据集根目录）

### MAT 文件内容

#### `multiview_result_<lf_name>.mat`

包含单个光场的所有超分结果：

**超分结果（4D 矩阵: [U, V, H, W]）：**
- `sr_aggre_sr`: Aggre 阶段 SR 结果
- `sr_aggre_ensemble`: Aggre 阶段 Ensemble 结果
- `sr_aggre_final`: Aggre 阶段 Final 结果
- `sr_ft_sr`: Finetune 阶段 SR 结果
- `sr_ft_ensemble`: Finetune 阶段 Ensemble 结果
- `sr_ft_final`: Finetune 阶段 Final 结果
- `vdsr_refs`: VDSR 参考结果

**PSNR 矩阵（2D 矩阵: [U, V]）：**
- `psnr_aggre_sr`: 每个视图的 Aggre SR PSNR
- `psnr_aggre_ensemble`: 每个视图的 Aggre Ensemble PSNR
- `psnr_aggre_final`: 每个视图的 Aggre Final PSNR
- `psnr_ft_sr`: 每个视图的 FT SR PSNR
- `psnr_ft_ensemble`: 每个视图的 FT Ensemble PSNR
- `psnr_ft_final`: 每个视图的 FT Final PSNR
- `psnr_vdsr`: 每个视图的 VDSR PSNR

**平均 PSNR（标量）：**
- `psnr_avg_aggre_sr`: Aggre SR 平均 PSNR
- `psnr_avg_aggre_ensemble`: Aggre Ensemble 平均 PSNR
- `psnr_avg_aggre_final`: Aggre Final 平均 PSNR
- `psnr_avg_ft_sr`: FT SR 平均 PSNR
- `psnr_avg_ft_ensemble`: FT Ensemble 平均 PSNR
- `psnr_avg_ft_final`: FT Final 平均 PSNR
- `psnr_avg_vdsr`: VDSR 平均 PSNR

**元信息：**
- `input_view_num`: 输入视图数量
- `target_view_range`: 目标超分区域大小
- `target_view_coords`: 目标视图坐标列表
- `psnr_bicubic`: Bicubic 插值基线 PSNR

#### `summary_all.mat`

包含所有光场的统计汇总：
- `lf_names`: 所有处理的光场名称列表
- `psnr_per_lf`: 每个光场的 PSNR 字典
- `average_psnr_aggre_final`: 总体平均 Aggre Final PSNR
- `average_psnr_ft_final`: 总体平均 FT Final PSNR
- `average_psnr_vdsr`: 总体平均 VDSR PSNR
- `average_psnr_bicubic`: 总体平均 Bicubic PSNR
- `config`: 训练配置信息

### 在 MATLAB 中读取结果

```matlab
% 读取单个光场结果
result = load('results/.../multiview_result_Bikes.mat');

% 查看超分结果（5x5x512x512）
sr_final = result.sr_ft_final;

% 查看 PSNR 矩阵（5x5）
psnr_matrix = result.psnr_ft_final;

% 显示中心视图
figure; imshow(squeeze(sr_final(3, 3, :, :)));

% 显示 PSNR 热图
figure; imagesc(psnr_matrix); colorbar;
title('PSNR per View');
```

## 训练时间估计

假设单个视图训练需要 T 分钟，则：
- 5×5 目标区域：约 25T 分钟
- 7×7 目标区域：约 49T 分钟
- 9×9 目标区域：约 81T 分钟

**建议**：
- 先用小数据集和小区域（如 5×5）测试
- 确认结果无误后再处理完整数据集

## 与原始单视图方法的对比

| 特性 | 单视图方法 | 多视图方法 |
|------|-----------|-----------|
| 输出 | 1 个视图 | 5×5 或 7×7 个视图 |
| 训练时间 | T | 25T 或 49T |
| 结果格式 | 2D 图像 | 4D 矩阵 |
| PSNR | 单个值 | 矩阵 + 平均值 |
| 适用场景 | 单一视图超分 | 多视图重建 |

## 注意事项

1. **显存占用**：每个视图独立训练，显存占用与单视图相同
2. **训练时间**：总时间 = 单视图时间 × 目标视图数量
3. **模型权重**：不同 `input_views` 需要重新训练（无法共享权重）
4. **数据要求**：输入光场必须包含足够的视图（≥ input_views）

## 常见问题

### Q1: 如何只超分中心视图？
使用原始的 `Main_LFZSSR.py` 即可。

### Q2: 可以超分非中心的区域吗？
可以，修改 `_compute_target_coords()` 方法中的起始坐标即可。

### Q3: 如何加速训练？
- 减少 `max_iter_aggre` 和 `max_iter_ft`
- 使用更少的目标视图
- 使用多 GPU（目前不支持，需要额外实现）

### Q4: 结果文件太大怎么办？
可以只保存关键结果（如 `sr_ft_final` 和 `psnr_ft_final`），修改 `_aggregate_results()` 方法。

## 技术细节

### 核心设计

`LFZSSR_MultiTargetView` 类是一个**包装器**，它：
1. 计算需要超分的目标视图坐标
2. 循环调用 `LFZSSR_SingleTargetView` 进行训练
3. 收集并整理所有视图的结果

**不重写任何训练逻辑**，完全复用现有代码。

### 配置更新

在循环中更新以下配置：
- `conf.view_num`: 输入视图数量
- `conf.refPos`: 当前目标视图坐标

这两个参数会传递给 Dataloader 和 Model，确保正确处理不同的视图。

## 许可与引用

如果您使用此代码，请引用原始 LFZSSR 论文和本实现。

## 联系方式

如有问题或建议，请提交 Issue。

