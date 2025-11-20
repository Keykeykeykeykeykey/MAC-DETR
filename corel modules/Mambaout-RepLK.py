class GatedUniRepLKBlock_BCHW(nn.Module):
    """
    门控 UniRepLKNet 块 (BCHW 格式)
    结合门控机制和 UniRepLKNet 的大核卷积
    参考: https://arxiv.org/pdf/1612.08083
    """
    def __init__(self, dim, expansion_ratio=8/3, kernel_size=7, conv_ratio=1.0,
                 norm_layer=None, act_layer=nn.GELU, drop_path=0.0):
        super().__init__()
        
        # 默认归一化层
        if norm_layer is None:
            norm_layer = partial(LayerNormGeneral, eps=1e-6, normalized_dim=(1, 2, 3))
        
        self.dim = dim
        self.expansion_ratio = expansion_ratio
        self.kernel_size = kernel_size
        self.conv_ratio = conv_ratio
        
        # 网络组件
        self.norm = norm_layer((dim, 1, 1))
        hidden_dim = int(expansion_ratio * dim)
        conv_channels = int(conv_ratio * dim)
        
        # 通道分割索引
        self.split_indices = (hidden_dim, hidden_dim - conv_channels, conv_channels)
        
        # 门控卷积块
        self.gate_conv = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1)
        self.activation = act_layer()
        
        # UniRepLKNet 卷积块
        self.unireplk_conv = UniRepLKNetBlock(conv_channels, kernel_size=kernel_size)
        
        # 输出投影
        self.output_proj = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        
        # Drop path 正则化
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        
        # 归一化
        x = self.norm(x)
        
        # 门控分割
        gate_output = self.gate_conv(x)
        g, i, c = torch.split(gate_output, self.split_indices, dim=1)
        
        # UniRepLKNet 卷积处理
        c = self.unireplk_conv(c)
        
        # 门控融合
        activated_gate = self.activation(g)
        fused_features = torch.cat([i, c], dim=1)
        modulated_features = activated_gate * fused_features
        
        # 输出投影
        x = self.output_proj(modulated_features)
        
        # 残差连接
        x = self.drop_path(x)
        x = x + shortcut
        
        return x

    def get_parameters_count(self):
        """返回参数数量"""
        return sum(p.numel() for p in self.parameters())


class MambaOut_RepLK(nn.Module):
    """
    MambaOut RepLK 模块
    基于 C2f 架构，使用门控 UniRepLKNet 块作为基本构建单元
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, 
                 expansion_ratio=8/3, kernel_size=7, conv_ratio=1.0):
        super().__init__()
        
        self.input_channels = c1
        self.output_channels = c2
        self.num_blocks = n
        self.use_shortcut = shortcut
        self.groups = g
        self.expansion_ratio = e
        
        # 计算隐藏层通道数
        self.hidden_channels = int(c2 * e)
        
        # 初始化 C2f 基类
        super(C2f, self).__init__()
        self._init_c2f_components()
        
        # 构建 MambaOut 块序列
        self.mamba_blocks = self._build_mamba_blocks(
            expansion_ratio, kernel_size, conv_ratio
        )

    def _init_c2f_components(self):
        """初始化 C2f 基础组件"""
        self.c = self.hidden_channels  # 隐藏通道数
        
        # 输入卷积
        self.cv1 = Conv(self.input_channels, 2 * self.c, 1, 1)
        
        # 输出卷积
        self.cv2 = Conv((2 + self.num_blocks) * self.c, self.output_channels, 1)
        
        # 如果没有块，则使用身份映射
        if self.num_blocks == 0:
            self.m = nn.Identity()

    def _build_mamba_blocks(self, expansion_ratio, kernel_size, conv_ratio):
        """构建 MambaOut 块序列"""
        return nn.ModuleList([
            GatedUniRepLKBlock_BCHW(
                dim=self.c,
                expansion_ratio=expansion_ratio,
                kernel_size=kernel_size,
                conv_ratio=conv_ratio
            ) for _ in range(self.num_blocks)
        ])

    def forward(self, x):
        """前向传播"""
        # C2f 标准前向传播逻辑
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.mamba_blocks)
        
        return self.cv2(torch.cat(y, 1))

    def switch_to_deploy(self):
        """切换到部署模式（如果支持）"""
        for block in self.mamba_blocks:
            if hasattr(block, 'switch_to_deploy'):
                block.switch_to_deploy()

    def get_output_shape(self, input_shape):
        """计算输出特征图形状"""
        batch_size, channels, height, width = input_shape
        return (batch_size, self.output_channels, height, width)

    def get_parameters_count(self):
        """返回总参数数量"""
        total_params = sum(p.numel() for p in self.parameters())
        mamba_params = sum(p.numel() for block in self.mamba_blocks for p in block.parameters())
        
        return {
            'total_parameters': total_params,
            'mamba_blocks_parameters': mamba_params,
            'other_parameters': total_params - mamba_params
        }

    def get_config(self):
        """返回模块配置"""
        return {
            'input_channels': self.input_channels,
            'output_channels': self.output_channels,
            'num_blocks': self.num_blocks,
            'hidden_channels': self.c,
            'use_shortcut': self.use_shortcut,
            'expansion_ratio': self.expansion_ratio
        }


class MambaOut_RepLK_Enhanced(MambaOut_RepLK):
    """
    增强版 MambaOut RepLK
    添加了额外的功能和配置选项
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5,
                 expansion_ratio=8/3, kernel_size=7, conv_ratio=1.0,
                 use_attention=False, drop_path_rate=0.0):
        
        self.use_attention = use_attention
        self.drop_path_rate = drop_path_rate
        
        super().__init__(c1, c2, n, shortcut, g, e, 
                        expansion_ratio, kernel_size, conv_ratio)

    def _build_mamba_blocks(self, expansion_ratio, kernel_size, conv_ratio):
        """构建增强的 MambaOut 块序列"""
        blocks = []
        for i in range(self.num_blocks):
            # 逐渐增加 drop path 率
            current_drop_path = self.drop_path_rate * i / max(self.num_blocks - 1, 1)
            
            block = GatedUniRepLKBlock_BCHW(
                dim=self.c,
                expansion_ratio=expansion_ratio,
                kernel_size=kernel_size,
                conv_ratio=conv_ratio,
                drop_path=current_drop_path
            )
            blocks.append(block)
        
        return nn.ModuleList(blocks)
