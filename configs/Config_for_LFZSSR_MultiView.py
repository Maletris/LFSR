"""
多视图光场超分辨率训练配置
继承自原始 Config 类，添加多视图相关参数
"""

from configs.Config_for_LFZSSR import Config as BaseConfig

class Config(BaseConfig):
    """
    多视图训练配置，继承原始配置
    """
    
    # 多视图专用参数
    input_view_num = 9      # 输入光场的视图数量（9/13/15）
    target_view_range = 5   # 目标超分区域大小（5表示5x5，7表示7x7）
    
    # 建议的配置覆盖
    record = True  # 多视图训练建议开启记录以便追踪每个视图的训练过程
    
    # 其他参数继承自父类 BaseConfig
    # 如需修改，可以在这里覆盖，例如：
    # max_iter_aggre = 2000  # 减少迭代次数以加快多视图训练
    # max_iter_ft = 2000

