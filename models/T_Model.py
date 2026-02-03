import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def spatial_fold(input, fold):
    """
    Downsamples the input by folding spatial dimensions into channels.
    """
    if fold == 1:
        return input

    batch, channel, height, width = input.shape
    h_fold = height // fold
    w_fold = width // fold

    return (
        input.view(batch, channel, h_fold, fold, w_fold, fold)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(batch, -1, h_fold, w_fold)
    )


def spatial_unfold(input, unfold):
    """
    Upsamples the input by unfolding channels into spatial dimensions.
    """
    if unfold == 1:
        return input

    batch, channel, height, width = input.shape
    h_unfold = height * unfold
    w_unfold = width * unfold

    return (
        input.view(batch, -1, unfold, unfold, height, width)
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(batch, -1, h_unfold, w_unfold)
    )

class UpTransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_planes, out_planes, 3, 1, 1),
            nn.GroupNorm(8, out_planes), nn.ReLU(inplace=True)
        )
        # Attention gate on skip (y) to suppress haze/noise
        self.gate = nn.Sequential(
            nn.Conv2d(out_planes + out_planes, out_planes, 1),
            nn.Sigmoid()
        )
        self.mix = nn.Sequential(
            nn.Conv2d(out_planes*2, out_planes, 3, 1, 1),
            nn.GroupNorm(8, out_planes), nn.ReLU(inplace=True)
        )
    def forward(self, y, x):
        x = self.up(x)
        y_resized = F.interpolate(y, size=x.shape[-2:], mode='bilinear', align_corners=True)
        w = self.gate(torch.cat([x, y_resized], dim=1))
        y_att = y_resized * w
        out = self.mix(torch.cat([x, y_att], dim=1))
        return out
   
class RDB_5C(nn.Module):
    """Residual Dense Block with 5 convolutional layers."""
    def __init__(self, nf, bias=True):
        super(RDB_5C, self).__init__()
        # Growth channel: intermediate channels
        self.conv1 = nn.Conv2d(nf, 32, 3, 1, 1, bias=bias)
        self.bn1 = nn.GroupNorm(8, 32)
        self.conv2 = nn.Conv2d(nf + 32, 32, 3, 1, 1, bias=bias)
        self.bn2 = nn.GroupNorm(8, 32)
        self.conv3 = nn.Conv2d(nf + 2 * 32, 32, 3, 1, 1, bias=bias)
        self.bn3 = nn.GroupNorm(8, 32)
        self.conv4 = nn.Conv2d(nf + 3 * 32, 32, 3, 1, 1, bias=bias)
        self.bn4 = nn.GroupNorm(8, 32)
        self.conv5 = nn.Conv2d(nf + 4 * 32, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        

    def forward(self, x):
        x1 = self.lrelu(self.bn1(self.conv1(x)))
        x2 = self.lrelu(self.bn2(self.conv2(torch.cat([x, x1], 1))))
        x3 = self.lrelu(self.bn3(self.conv3(torch.cat([x, x1, x2], 1))))
        x4 = self.lrelu(self.bn4(self.conv4(torch.cat([x, x1, x2, x3], 1))))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5 + x
    
class REC(nn.Module):
    def __init__(self, num_init_features, out_c):
        super(REC, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(num_init_features, num_init_features//2, kernel_size=1, stride=1, padding=0, bias=True)
        # self.bn1 = nn.BatchNorm2d(num_init_features//2)
        self.conv1 = nn.Conv2d(num_init_features//2, num_init_features//2, kernel_size=1, stride=1, padding=0, bias=True)
        # self.bn2 = nn.BatchNorm2d(num_init_features//2)
        self.deconv2 = nn.ConvTranspose2d(num_init_features//2, out_c, kernel_size=1, stride=1, padding=0, bias=True)
        # self.bn3 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=1, stride=1, padding=0, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
    def forward(self, x):
        x = self.lrelu(self.deconv1(x))
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.deconv2(x))
        x = self.conv2(x)
        return x

class ASPP(nn.Module):
    def __init__(self, in_channel=6, out_channel=16):
        super(ASPP, self).__init__()
        self.pooling = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Conv2d(in_channel, out_channel, 1, 1),
                    nn.GroupNorm(8, out_channel),
                    nn.ReLU(inplace=True))

        self.atrous_block1 = nn.Sequential( nn.Conv2d(in_channel, out_channel, 1, 1),
                                          nn.GroupNorm(8, out_channel),
                                          nn.ReLU(inplace=True))
        
        self.atrous_block6 = nn.Sequential( nn.Conv2d(in_channel, out_channel, 3, 1, padding=4, dilation=4),
                                          nn.GroupNorm(8, out_channel),
                                          nn.ReLU(inplace=True))
        
        self.atrous_block12 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, padding=8, dilation=8),
                                          nn.GroupNorm(8, out_channel),
                                          nn.ReLU(inplace=True))
        self.atrous_block18 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, padding=12, dilation=12),
                                          nn.GroupNorm(8, out_channel),
                                          nn.ReLU(inplace=True))

        self.output = nn.Sequential(
            nn.Conv2d(5 * out_channel, out_channel, 1, bias=False),
            nn.GroupNorm(8, out_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)


    def forward(self, x):

        image_features = self.pooling(x)
        image_features = F.interpolate(image_features, size=x.size()[2:], mode='bilinear',align_corners=True)
        
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        net = self.output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net

def window_partition(x, window_size):
    """
    Partitions the input (B, C, H, W) into windows -> (num_windows * B, window_size, window_size, C).
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    windows = x.view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Reverses the partitioned windows back to original image -> (B, C, H, W).
    """
    B = int(windows.shape[0] / (H // window_size * W // window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    return x.view(B, -1, H, W)

# ---------------- DropPath ----------------
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

# ---------------- Frequency Domain Window Attention ----------------
class WindowAttentionFreq(nn.Module):
    """
    Performs attention in the frequency domain within each window.
    Workflow: 2D-FFT -> Split complex to real/imag -> Self-attention in freq domain ->
    Reconstruct complex -> iFFT -> Return spatial domain output.
    """
    def __init__(self, in_channels, num_heads, window_size=8,
                 qkv_bias=True, attn_drop=0., proj_drop=0., keep_ratio=0.30):
        """
        in_channels: Source channels C (internally merges real/imag to 2C).
        """
        super(WindowAttentionFreq, self).__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.num_heads = num_heads
        self.freq_dim = in_channels * 2  # real + imag
        head_dim = self.freq_dim // num_heads
        assert head_dim * num_heads == self.freq_dim, "freq_dim must be divisible by num_heads"
        self.scale = head_dim ** -0.5

        # qkv / proj defined on freq_dim (real+imag)
        self.qkv = nn.Linear(self.freq_dim, self.freq_dim * 3, bias=qkv_bias)
        self.pe = nn.Sequential(
            nn.Linear(2, self.freq_dim // 2), nn.GELU(),
            nn.Linear(self.freq_dim // 2, self.freq_dim)
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.freq_dim, self.freq_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.keep_ratio = keep_ratio
        self.lp_gate = nn.Parameter(torch.tensor(0.8))  # Low-frequency fidelity gate
        self.hp_gate = nn.Parameter(torch.tensor(0.6))  # High-frequency enhancement gate

    def low_high_masks(self, win, device):
        K = max(1, int(win * self.keep_ratio))
        mask = torch.zeros(win, win, device=device)
        mask[:K, :K] = 1.0
        lp = mask
        hp = 1.0 - lp
        return lp[None,None], hp[None,None]  # (1,1,win,win)

    def forward(self, x_windows):
        """
        x_windows: (nW*B, win, win, C) spatial real input in each window
        returns: (nW*B, win, win, C) spatial real output in each window
        """
        nW_B, win, _, C = x_windows.shape
        # Permute for fft2 on spatial dims
        x_perm = x_windows.permute(0, 3, 1, 2).contiguous()  # (nW_B, C, win, win)

        # 1) FFT2 -> Complex spectrum (nW_B, C, win, win)
        Xf = torch.fft.fft2(x_perm, dim=(-2, -1))
        # Mask calculation for low/high frequency split
        K = max(1, int(win * self.keep_ratio))
        lp = torch.zeros(win, win, device=Xf.real.device)
        lp[:K, :K] = 1.0
        lp = lp[None, None]  # Broadcast to (1, 1, win, win)
        hp = 1.0 - lp

        # Low-frequency fidelity path
        Xf_lp = Xf * lp

        # High-frequency attention enhancement path
        X_real, X_imag = (Xf.real * hp), (Xf.imag * hp)

        # 2) Concatenate real/imag for real-valued attention features
        X_cat = torch.cat([X_real, X_imag], dim=1)  # (nW_B, 2C, win, win)

        # 3) As tokens: (nW_B, win*win, 2C)
        tokens = X_cat.permute(0, 2, 3, 1).contiguous().view(-1, win * win, self.freq_dim)

        uu, vv = torch.meshgrid(torch.linspace(0,1,win, device=tokens.device),
                        torch.linspace(0,1,win, device=tokens.device), indexing='ij')
        freq_xy = torch.stack([uu, vv], dim=-1).view(1, win*win, 2).repeat(tokens.size(0),1,1)
        tokens = tokens + self.pe(freq_xy)

        # 4) qkv & attention in frequency-coefficient positions
        qkv = self.qkv(tokens).reshape(-1, win * win, 3, self.num_heads, self.freq_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # shapes: (3, nW_B, num_heads, N, head_dim) after permute indexing

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(-1, win * win, self.freq_dim)  # (nW_B, N, 2C)
        out = self.proj_drop(self.proj(out))

        # 6) Split real/imag, reconstruct complex spectrum, and iFFT
        out_real = out_cat[:, :C, :, :]
        out_imag = out_cat[:, C:, :, :]
        Xf_hp_enh = torch.complex(out_real, out_imag)

        # Combine low-frequency fidelity and high-frequency enhancement
        Xf_out = self.lp_gate * Xf_lp + self.hp_gate * Xf_hp_enh

        # Inverse FFT to spatial domain (take real part as output)
        x_spatial = torch.fft.ifft2(Xf_out, dim=(-2, -1)).real  # (nW_B, C, win, win)

        # Final shape (nW_B, win, win, C)
        x_spatial = x_spatial.permute(0, 2, 3, 1).contiguous()

        return x_spatial


# ---------------- MLP Feedforward ----------------
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, drop=0.):
        super(MLP, self).__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

# ---------------- Swin Transformer Block (Frequency Domain Attention) ----------------
class SwinBlockFreq(nn.Module):
    """
    Swin Block but attention is performed in frequency domain inside each window.
    Input/Output: (B, C, H, W)
    """
    def __init__(self, dim, num_heads=4, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super(SwinBlockFreq, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.pre_bn = nn.GroupNorm(8, dim)
        self.norm1 = nn.LayerNorm(dim)
        # WindowAttentionFreq expects original C (in_channels)
        self.attn = WindowAttentionFreq(in_channels=dim, num_heads=num_heads, window_size=window_size,
                                        qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        self.conv_branch = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim), 
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, 1, 0)
         )
        self.mix_gate_logit = nn.Parameter(torch.zeros(1, dim, 1, 1))
        
        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),  # depthwise
            nn.GroupNorm(8, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, 1, 0)                    # pointwise
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)
    
    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, C, H, W)
        """
        B, C, H, W = x.shape
        assert C == self.dim, "channel dim mismatch"
        shortcut = x

        # Pre-norm and spatial to token conversion
        x_bn = self.pre_bn(x)
        x_tokens = x_bn.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x_tokens = self.norm1(x_tokens)
        x_pre = x_tokens.transpose(1, 2).view(B, C, H, W)  # (B, C, H, W)

        # Shift if required
        if self.shift_size > 0:
            x_shift = torch.roll(x_pre, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            x_shift = x_pre

        # Partition windows
        x_windows = window_partition(x_shift, self.window_size)  # (nW*B, win, win, C)

        # Frequency-domain attention inside windows
        attn_windows = self.attn(x_windows)  # (nW*B, win, win, C) spatial outputs

        # Reverse windows -> (B, C, H, W)
        x_attn = window_reverse(attn_windows, self.window_size, H, W)

        x_conv = self.conv_branch(x_shift)  # (B,C,H,W)

        # Reverse shift if applied
        if self.shift_size > 0:
            x_attn = torch.roll(x_attn, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
            x_conv = torch.roll(x_conv, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        # Gated mixture fusion
        gate = torch.sigmoid(self.mix_gate_logit)
        # Broadcast gate to (1, C, 1, 1) if necessary
        x_out = gate * x_attn + (1.0 - gate) * x_conv

        x_out = self.refine(x_out)
        x = shortcut + self.drop_path(x_out)

        # MLP block (pre-norm then mlp) -- operate on spatial tokens
        shortcut2 = x
        x2 = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x2 = self.norm2(x2)
        x2 = self.mlp(x2)
        x = shortcut2 + self.drop_path(x2.transpose(1, 2).view(B, C, H, W))

        return x

class FinalHead(nn.Module):
    def __init__(self, in_ch=64+128+256, mid=128, out_ch=1):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, 1, 1), nn.GroupNorm(8, mid), nn.GELU(),
            nn.Conv2d(mid, out_ch, 1)
        )
    def forward(self, f1, f2, f3):  # f1:64x, f2:128x, f3:256x decoder feats
        f2 = F.interpolate(f2, size=f1.shape[-2:], mode='bilinear', align_corners=True)
        f3 = F.interpolate(f3, size=f1.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([f1, f2, f3], dim=1)
        return self.fuse(x)

class T_network(nn.Module):
    def __init__(self, n_classes=2):
        super(T_network,self).__init__()

        T_resnet = models.resnet101(pretrained=True)
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
#        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        
        self.encoder1 = T_resnet.layer1  #[2, 256, 256, 256]
        self.encoder2 = T_resnet.layer2  #[2, 512, 128, 128]
        self.encoder3 = T_resnet.layer3  #[2, 1024, 64, 64]
        self.encoder4 = T_resnet.layer4  #[2, 2048, 32, 32]
        
        self.just1 = nn.Sequential(
            nn.Conv2d(256, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        
        self.just2 = nn.Sequential(
            nn.Conv2d(512, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        
        self.just3 = nn.Sequential(
            nn.Conv2d(1024, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        
        self.aspp = ASPP(2048, 512) #only change channel
        
        self.up1 = UpTransitionBlock(512, 256)
        self.decoder1 = RDB_5C(256)#RRDB(256)#
        self.att1 = SwinBlockFreq(256)
        self.out1 = REC(256, 4)
        
        self.up2 = UpTransitionBlock(256, 128)
        self.decoder2 = RDB_5C(128)#RRDB(128)#
        self.att2 = SwinBlockFreq(128)
        self.out2 = REC(128, 4)
        
        self.up3 = UpTransitionBlock(128, 64)
        self.decoder3 =RDB_5C(64)# RRDB(64)#
        self.att3 = SwinBlockFreq(64)
        self.out3 = REC(64, 4)
        
        self.final = FinalHead(in_ch=64+128+256, mid=128, out_ch=1)
        
    def forward(self, x):#512*512

        # hx = spatial_fold(x, 2)

        ## -------------Encoder-------------
        hx =self.relu1(self.bn1(self.conv1(x)))

        h1 = self.encoder1(hx) # 256
        h2 = self.encoder2(h1) # 128
        h3 = self.encoder3(h2) # 64
        h4 = self.encoder4(h3) # 32
        
        h1 = self.just1(h1) #torch.Size([2, 64, 256, 256])
        h2 = self.just2(h2) #torch.Size([2, 128, 128, 128])
        h3 = self.just3(h3) #torch.Size([2, 256, 64, 64])
        h4 = self.aspp(h4) #torch.Size([2, 512, 32, 32])
        
        up1 = self.att1(self.up1(h3, h4))
        d1 = self.decoder1(up1)
        out1 = self.out1(d1) #[2, 12, 64, 64]
        out1 = spatial_unfold(out1, 2)#[2, 1, 128, 128]
#        print(d1.size())  # 64
        
        up2 = self.att2(self.up2(h2, d1))
        d2 = self.decoder2(up2)
        out2 = self.out2(d2)
        out2 = spatial_unfold(out2, 2)#[2, 1, 256, 256]
        
        up3 = self.att3(self.up3(h1, d2))
        d3 = self.decoder3(up3)
        out3 = self.out3(d3)
        out3 = spatial_unfold(out3, 2)#[2, 1, 512, 512]
        
        final_mask = self.final(d3, d2, d1)
        final_mask = torch.sigmoid(F.interpolate(final_mask, size=x.size()[2:], mode='bilinear', align_corners=True))

        return torch.sigmoid(out1), torch.sigmoid(out2), torch.sigmoid(out3), final_mask, h4, h3, h2, h1

if __name__ == "__main__":
    model1 = T_network().cuda()
    x = torch.randn(2, 3, 512, 512).cuda()
    out1, out2, out3, h4, h3, h2, h1 = model1(x)
    print(f"Output scales sizes: {out1.size()}, {out2.size()}, {out3.size()}")

#    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model1.parameters())))
##    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model1.parameters())))
#    print(out1234.size())
#    haze_class = models.resnet18()
##    densenet121(pretrained=True)
##    conv0=haze_class.features.conv0
#    print(haze_class)
#    
    
    
#    










