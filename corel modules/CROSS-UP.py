class CrossChannelShift(nn.Module):
    """
    交叉通道移位模块
    对输入通道进行分组并在不同方向上进行移位操作
    """
    def __init__(self, shift_size=1):
        super(CrossChannelShift, self).__init__()
        self.shift_size = shift_size

    def forward(self, x):
        # 将通道分为4组
        x1, x2, x3, x4 = x.chunk(4, dim=1)
        
        # 在四个不同方向上进行移位操作
        x1 = torch.roll(x1, shifts=self.shift_size, dims=2)   # 向下移位
        x2 = torch.roll(x2, shifts=-self.shift_size, dims=2)  # 向上移位
        x3 = torch.roll(x3, shifts=self.shift_size, dims=3)   # 向右移位
        x4 = torch.roll(x4, shifts=-self.shift_size, dims=3)  # 向左移位
        
        # 重新拼接通道
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return x


class CROSS_UP(nn.Module):
    """
    交叉上采样模块
    结合上采样、深度卷积和交叉通道移位操作
    """
    def __init__(self, in_channels, kernel_size=3, stride=1, shift_size=1):
        super(CROSS_UP, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = in_channels
        
        # 上采样和深度可分离卷积
        self.upsample_dw_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv(
                in_channels=in_channels,
                out_channels=in_channels,
                k=kernel_size,
                g=in_channels,  # 深度可分离卷积
                s=stride
            )
        )
        
        # 点卷积 (1x1卷积)
        self.pointwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        
        # 交叉通道移位模块
        self.cross_shift = CrossChannelShift(shift_size=shift_size)

    def forward(self, x):
        # 上采样和深度卷积
        x = self.upsample_dw_conv(x)
        
        # 通道混洗和交叉移位
        x = self._apply_cross_channel_operation(x)
        
        # 点卷积融合特征
        x = self.pointwise_conv(x)
        
        return x

    def _apply_cross_channel_operation(self, x):
        """
        应用通道混洗和交叉移位操作
        """
        batch_size, num_channels, height, width = x.size()
        
        # 确保通道数可以被4整除
        if num_channels % 4 != 0:
            raise ValueError(f"Channel number {num_channels} must be divisible by 4")
        
        groups = 4
        channels_per_group = num_channels // groups
        
        # 通道混洗操作
        # 1. 重塑为 [batch, groups, channels_per_group, height, width]
        x = x.view(batch_size, groups, channels_per_group, height, width)
        
        # 2. 转置维度 [batch, channels_per_group, groups, height, width]
        x = torch.transpose(x, 1, 2).contiguous()
        
        # 3. 重塑回原始形状 [batch, num_channels, height, width]
        x = x.view(batch_size, -1, height, width)
        
        # 4. 应用交叉通道移位
        x = self.cross_shift(x)
        
        return x

    def get_output_shape(self, input_shape):
        """
        计算输出特征图形状
        """
        batch_size, channels, height, width = input_shape
        output_height = height * 2
        output_width = width * 2
        return (batch_size, self.out_channels, output_height, output_width)
