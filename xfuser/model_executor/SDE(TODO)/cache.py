
class CacheContext:
    """通用缓存上下文，存储所有模型的中间状态和统计信息"""
    def __init__(self):
        # 特征缓存
        self.hidden_states = None  # 当前特征
        self.prev_hidden_states = None  # 上一步特征（用于变化率计算）
        self.encoder_hidden_states = None  # 编码器特征
        self.prev_encoder_hidden_states = None  # 上一步编码器特征
        # 注意力缓存
        self.attention_maps = None  # 当前注意力图
        self.prev_attention_maps = None  # 上一步注意力图
        # 残差缓存（复用缓存时的补偿值）
        self.hidden_residual = None
        self.encoder_residual = None
        # 统计信息
        self.cache_hits = 0
        self.total_steps = 0