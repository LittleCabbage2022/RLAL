# src/utils/flops_utils.py

def calculate_conv_flops(h_in, w_in, k, stride, c_in, c_out):
    """
    计算单个卷积层的 FLOPs
    FLOPs = H_out * W_out * K * K * C_in * C_out
    (忽略 Bias 项，因其占比极小)
    """
    h_out = h_in // stride
    w_out = w_in // stride
    return h_out * w_out * k * k * c_in * c_out

def calculate_block_flops(info, current_cin):
    """
    计算整个 Block (Conv1 + Conv2) 的 FLOPs
    """
    # Conv1: Cin -> Cout1 (Pruned dim)
    f1 = calculate_conv_flops(info['h1'], info['w1'], info['k1'], info['s1'], 
                              current_cin, info['cout1'])
    # Conv2: Cout1 -> Cout2
    f2 = calculate_conv_flops(info['h2'], info['w2'], info['k2'], info['s2'], 
                              info['cout1'], info['cout2'])
    return f1 + f2

def count_total_flops(model_wrapper):
    """计算当前模型配置下的总 FLOPs (基于原始通道数)"""
    total = 0.0
    for i in range(model_wrapper.get_num_prunable_layers()):
        info = model_wrapper.get_layer_static_info(i)
        # 初始状态下，Block 输入通道 = info['cin1']
        total += calculate_block_flops(info, info['cin1'])
    return total

# def calculate_conv_flops(h_in, w_in, k, stride, c_in, c_out):
#     """
#     计算单个卷积层的 FLOPs
#     FLOPs = H_out * W_out * K * K * C_in * C_out
#     (忽略 Bias 项，因其占比极小)
#     """
#     h_out = h_in // stride # 简化计算，假设 padding='same' 或合适
#     w_out = w_in // stride
    
#     flops = h_out * w_out * k * k * c_in * c_out
#     return flops

# def count_total_flops(model_wrapper):
#     """计算当前模型配置下的总 FLOPs (基于原始通道数)"""
#     total_flops = 0.0
#     for i in range(model_wrapper.get_num_prunable_layers()):
#         info = model_wrapper.get_layer_static_info(i)
#         f = calculate_conv_flops(
#             info['h'], info['w'], info['k'], info['s'], 
#             info['cin'], info['cout']
#         )
#         total_flops += f
#     return total_flops