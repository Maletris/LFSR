"""
Parameters for LFZSSR.
"""

class Config:

    scale = 2
    view_num = 9
    refPos = [4, 4]
    patch_size = 64
    align_patch_size = 64
    weight_decay = 0.0
    weight_decay_aggre = 0.0
    aggre_batch_size = 1
    align_batch_size = 1
    ft_batch_size = 1
    random_seed = None
    data_range = "auto"  # infer 255/65535 from input dtype, or set a number explicitly
    result_dtype = "auto"  # infer from input dtype, or set "uint8"/"uint16"

    ######## devices
    use_cuda = True
    gpu_id = '0'

    ######## early stop and test
    max_iters = 20000
    test_step = 50
    min_learning_rate = 1e-6

    ######## record
    record = False
    display_loss_step = 20

    ######## For back-projection refinement
    max_bp_iter = 10
    scale_aug = True
    pad_size = 12

    ############### -----------------------

    # For VDSR
    vdsr_model_path = './pretrained/VDSR_model.pth'

    # For AlignNet
    disp_max = 2.0
    level_num = 64
    align_loss_type = "charbonnier"  # "charbonnier", "huber", or "mse"
    charbonnier_eps = 1e-3
    huber_delta = 1e-2
    residual_weight_alpha = 10.0
    residual_weight_min = 0.1
    disp_hessian_weight = 1e-3
    edge_weight_alpha = 10.0

    # For finetune
    align_loss_weight = 0.1
    set_name = "low" # "high" or "low"
    zssr_bp_ratio = 0.5

    # for scheduler learning
    lr_align_stage = 1e-4
    lr_aggre_stage = 1e-4
    lr_ft_stage = 1e-4

    max_iter_aggre = 3500
    max_iter_ft = 3500
    align_aggre_iter_step = 3000
    ft_iter_step = 2500

    # For AggreNet confidence weighting before fusion
    aggre_confidence_enable = True
    aggre_confidence_alpha = 10.0
    aggre_confidence_min = 0.1
    aggre_angular_weight_enable = False
    aggre_angular_weight_beta = 0.1
