"""
Multi-view Light Field Super-Resolution Main Function

Supports different input view numbers (9x9, 13x13, 15x15) and different target SR regions (5x5, 7x7).
"""
import sys
sys.path.append('./tools')
import argparse
import scipy.io as sio
from tools.utils import *
from trainers.Class_LFZSSR import LFZSSR_MultiTargetView
from configs.Config_for_LFZSSR_MultiView import Config

parser = argparse.ArgumentParser(description='Multi-view Light Field Super-Resolution Training')
parser.add_argument("--gpus", default="1", type=str, help="GPU ID (default: 0)")
parser.add_argument("--record", action="store_true", help="Record training logs")
parser.add_argument("--scale", default=3, type=int, help="Scaling factor (default: 2)")
parser.add_argument("--dataset", default="sLF_sparseBeads_9_Scale3", type=str, 
                    help="Dataset name: ")
parser.add_argument("--start", default=0, type=int, help="Start index in dataset (default: 0)")
parser.add_argument("--end", default=1, type=int, help="End index in dataset (default: 1)")
parser.add_argument("--gpu-num", default=1, type=int, help="Number of GPUs for batch size setting")

# Multi-view specific parameters
parser.add_argument("--input-views", default=9, type=int, 
                    choices=[9, 13, 15], 
                    help="Input light field view number (9/13/15, default: 9)")
parser.add_argument("--target-views", default=5, type=int,
                    choices=[5, 7, 9], 
                    help="Target super-resolution region size (5/7/9, default: 5)")

opt = parser.parse_args()

batch_size = opt.gpu_num
print("\n" + "="*70)
print("Multi-view Light Field Super-Resolution Training Parameters:")
print("="*70)
print(f"Dataset: {opt.dataset}")
print(f"Scaling Factor: {opt.scale}x")
print(f"Input Views: {opt.input_views}x{opt.input_views}")
print(f"Target Region: {opt.target_views}x{opt.target_views}")
print(f"GPU: {opt.gpus} (Number: {opt.gpu_num})")
print(f"Processing Range: {opt.start} to {opt.end}")
print("="*70 + "\n")

# 数据集配置
lf_set_name = opt.dataset
list_file_path = "./data/{}_list.txt".format(lf_set_name)

# Read data list
try:
    fd = open(list_file_path, 'r')
    name_list = [line.strip('\n') for line in fd.readlines()]
    fd.close()
    print(f"Successfully loaded data list: {list_file_path}")
    print(f"Dataset contains {len(name_list)} light fields: {name_list}\n")
except FileNotFoundError:
    print(f"Error: Cannot find data list file {list_file_path}")
    print("Please make sure the data list file exists!")
    exit(1)

# 配置超参数
save_name_dir = "LFZSSR_MultiView_Set_{}_Scale_{}_Input{}x{}_Target{}x{}".format(
    opt.dataset, opt.scale, opt.input_views, opt.input_views, 
    opt.target_views, opt.target_views)
save_prefix = "./results/{}".format(save_name_dir)

# 创建保存目录
os.makedirs(save_prefix, exist_ok=True)

configs = Config()

# 固定参数
configs.gpu_id = opt.gpus
configs.align_patch_size = 32
configs.aggre_batch_size = batch_size
configs.align_batch_size = batch_size
configs.ft_batch_size = batch_size
configs.zssr_bp_ratio = 0.5
configs.align_loss_weight = 0.1

# 多视图参数
configs.input_view_num = opt.input_views
configs.target_view_range = opt.target_views

# 视图配置
configs.view_num = opt.input_views
cv_uv = configs.view_num // 2
configs.refPos = [cv_uv, cv_uv]  # 初始值，会在 MultiTargetView 中更新
configs.record = opt.record
configs.level_num = 50

# Adjust parameters based on scaling factor
if opt.scale == 2:
    configs.disp_max = 2.0
    configs.scale = opt.scale
    configs.patch_size = 64
elif opt.scale == 3:
    configs.disp_max = 1.5
    configs.scale = opt.scale
    configs.patch_size = 72
else:
    raise Exception("Unsupported scaling factor! Please use 2 or 3")

# 特殊数据集配置
if lf_set_name == "HCI1":
    configs.set_name = 'high'
else:
    configs.set_name = 'low'

# Store statistical results for all light fields
all_lf_psnr = {}

##################### Start Processing Light Fields #####################

for ind in range(opt.start, opt.end):
    lf_name = name_list[ind]
    print("\n" + "="*70)
    print(f"Processing Light Field: {lf_name} ({ind+1-opt.start}/{opt.end-opt.start})")
    print("="*70 + "\n")
    
    mat_path = "./data/{}/{}.mat".format(lf_set_name, lf_name)
    save_name = lf_name
    
    # Create separate folder for each light field
    lf_save_prefix = os.path.join(save_prefix, lf_name)
    os.makedirs(lf_save_prefix, exist_ok=True)
    
    # Check if file exists
    if not os.path.exists(mat_path):
        print(f"Warning: Cannot find file {mat_path}, skipping...")
        continue
    
    # Calculate Bicubic baseline (using center view)
    try:
        hr_lf = loadmat(mat_path)["lf_hr"]
        # Calculate center view index based on actual light field size
        lf_size = hr_lf.shape[0]  # Assuming square angular resolution
        center_idx = lf_size // 2
        hr_cv = hr_lf[center_idx, center_idx]
        lr_cv = single_image_downscale(hr_cv, opt.scale,
                                       data_range=configs.data_range,
                                       result_dtype=configs.result_dtype)
        bic_lr_cv = single_image_upscale(lr_cv, opt.scale,
                                         data_range=configs.data_range,
                                         result_dtype=configs.result_dtype)
        psnr_bicubic = PSNR(bic_lr_cv, hr_cv, data_range=configs.data_range)
        print(f"Bicubic interpolation baseline PSNR: {psnr_bicubic:.4f} dB (center view [{center_idx}, {center_idx}])\n")
    except Exception as e:
        print(f"Warning: Cannot calculate Bicubic baseline: {e}")
        psnr_bicubic = 0.0
    
    # Create multi-view trainer
    multi_trainer = LFZSSR_MultiTargetView(
        lf_name=lf_name,
        mat_path=mat_path,
        conf=configs,
        save_name=save_name,
        save_prefix=lf_save_prefix,  # Use light field specific save path
        input_view_num=opt.input_views,
        target_view_range=opt.target_views,
        gpu_num=opt.gpu_num
    )
    
    # Run multi-view training
    results_dict = multi_trainer.run()
    
    # Add Bicubic baseline to results
    results_dict["psnr_bicubic"] = psnr_bicubic
    
    # Save results to MAT file (in light field folder)
    result_file = "{}/multiview_result_{}.mat".format(lf_save_prefix, lf_name)
    sio.savemat(result_file, results_dict)
    print(f"\nResults saved to: {result_file}")
    
    # Record statistical information
    all_lf_psnr[lf_name] = {
        "psnr_aggre_final": results_dict["psnr_avg_aggre_final"],
        "psnr_ft_final": results_dict["psnr_avg_ft_final"],
        "psnr_vdsr": results_dict["psnr_avg_vdsr"],
        "psnr_bicubic": psnr_bicubic,
    }
    
    print("\n" + "="*70)
    print(f"Light field {lf_name} processing completed!")
    print("="*70 + "\n")

# Save statistical results for all light fields
if len(all_lf_psnr) > 0:
    summary_file = "{}/summary_all.mat".format(save_prefix)
    
    # Calculate overall average PSNR
    avg_aggre_final = np.mean([v["psnr_aggre_final"] for v in all_lf_psnr.values()])
    avg_ft_final = np.mean([v["psnr_ft_final"] for v in all_lf_psnr.values()])
    avg_vdsr = np.mean([v["psnr_vdsr"] for v in all_lf_psnr.values()])
    avg_bicubic = np.mean([v["psnr_bicubic"] for v in all_lf_psnr.values()])
    
    summary = {
        "lf_names": list(all_lf_psnr.keys()),
        "psnr_per_lf": all_lf_psnr,
        "average_psnr_aggre_final": avg_aggre_final,
        "average_psnr_ft_final": avg_ft_final,
        "average_psnr_vdsr": avg_vdsr,
        "average_psnr_bicubic": avg_bicubic,
        "config": {
            "dataset": opt.dataset,
            "scale": opt.scale,
            "input_views": opt.input_views,
            "target_views": opt.target_views,
        }
    }
    
    sio.savemat(summary_file, summary)
    
    print("\n" + "="*70)
    print("All light fields processing completed!")
    print("="*70)
    print(f"Overall Average PSNR:")
    print(f"  - Bicubic:     {avg_bicubic:.4f} dB")
    print(f"  - VDSR:        {avg_vdsr:.4f} dB")
    print(f"  - Aggre Final: {avg_aggre_final:.4f} dB")
    print(f"  - FT Final:    {avg_ft_final:.4f} dB")
    print(f"\nSummary saved to: {summary_file}")
    print("="*70 + "\n")
else:
    print("\nWarning: No light fields were successfully processed!")

print("Program execution completed!")

