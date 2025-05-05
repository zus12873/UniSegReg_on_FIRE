import torch
import numpy as np

def fix_tensor_format(x):
    """
    修复张量格式，确保输入格式为 [batch, 1, height, width]
    处理可能的 [batch, 1, height, width, channels] 问题，并将RGB转换为灰度
    """
    # 如果是 numpy 数组，先转为 torch tensor
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    
    # 检查维度
    if len(x.shape) == 5 and x.shape[1] == 1 and x.shape[4] == 3:
        # 处理 [batch, 1, height, width, 3] 格式
        # 删除多余的维度，重排通道位置
        x = x.squeeze(1)  # 移除第二维 -> [batch, height, width, 3]
        
        # 将RGB转换为灰度 (使用标准RGB权重)
        # Y = 0.299 R + 0.587 G + 0.114 B
        rgb_weights = torch.tensor([0.299, 0.587, 0.114], device=x.device)
        # 对最后一个维度（RGB通道）做加权和
        x_gray = torch.matmul(x, rgb_weights)
        # 添加通道维度
        x = x_gray.unsqueeze(1)  # [batch, 1, height, width]
        
    elif len(x.shape) == 4 and x.shape[3] == 3:
        # 处理 [batch, height, width, 3] 格式
        # 将RGB转换为灰度
        rgb_weights = torch.tensor([0.299, 0.587, 0.114], device=x.device)
        x_gray = torch.matmul(x, rgb_weights)
        # 添加通道维度
        x = x_gray.unsqueeze(1)  # [batch, 1, height, width]
        
    # 如果已经是正确格式则不变
    
    return x 