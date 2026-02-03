import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from T_Model import spatial_fold, spatial_unfold, REC, ASPP, RDB_5C
from MyModules import DynamicMutualEnhancement, SpectrumAwareAggregation, ContextAwareAggregation
from utils.dcn import DeformableConv2d as DeformConv2d


class ResidualDenseBlock_3C(nn.Module):
    """Residual Dense Block with 3 convolutional layers."""
    def __init__(self, nf, bias=True):
        super(ResidualDenseBlock_3C, self).__init__()
        # growth channel: intermediate channels
        self.conv1 = nn.Conv2d(nf, 32, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + 32, 32, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * 32, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.conv3(torch.cat([x, x1, x2], 1))
        return x3 + x

class RRDB2(nn.Module):
    '''Residual in Residual Dense Block'''
    def __init__(self, nf):
        super(RRDB2, self).__init__()

        self.conv1 = nn.Conv2d(nf, nf, 1, stride=1, padding=0, bias=True)
        self.RDB1 = ResidualDenseBlock_3C(nf)
        self.RDB2 = ResidualDenseBlock_3C(nf)
        self.RDB3 = ResidualDenseBlock_3C(nf)
        self.RDB4 = ResidualDenseBlock_3C(nf)
        self.RDB5 = ResidualDenseBlock_3C(nf)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        out1 = self.RDB1(x1)
        out1 = self.RDB2(out1)
        out1 = self.RDB3(out1)
        out1 = self.RDB4(out1)
        out1 = self.RDB5(out1)
        out1 = out1 + x
        return out1

class Fusion_new(nn.Module):
    def __init__(self, in_channel=64, out_channels=32, expansion_factor=2, bias=True):
        super(Fusion_new, self).__init__()
        
        hidden_dim = int(out_channels * expansion_factor)
        self.in_proj = nn.Conv2d(in_channel, out_channels, kernel_size=1, bias=bias)
        
        # Spatial branch
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=bias),
            nn.GELU(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=5, padding=2, groups=out_channels, bias=bias)
        )
        self.dwconv = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, groups=in_channel, bias=bias)  
        
        # self.ca = ChannelAttention(out_channels)
        
        # Spectral branch - complex convolution
        self.spectral_real = nn.Sequential(
            nn.Conv2d(out_channels, hidden_dim, kernel_size=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=bias)
        )
        self.spectral_imag = nn.Sequential(
            nn.Conv2d(out_channels, hidden_dim, kernel_size=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=bias)
        )
        
        self.outconv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=bias)
        
    def forward(self, x_d, y_e, up=None):
        """
        Fused spatial and spectral features.
        """
        # Input shape: [B, C, H, W]
        if up is not None:
            x_d = F.interpolate(x_d, scale_factor=2, mode='bilinear', align_corners=False)
            x_d = self.in_proj(x_d)

        de = torch.cat([x_d, y_e], dim=1)

        # Spatial branch
        spatial_output = self.spatial_branch(de)
        spatial_out = self.dwconv(spatial_output)
        x, z = spatial_out.chunk(2, dim=1)

        # Spectral branch
        x_freq = torch.fft.rfft2(x, norm="ortho")
        # Process real and imaginary parts separately
        real_part = self.spectral_real(x_freq.real)
        imag_part = self.spectral_imag(x_freq.imag)

        # Combine back to complex
        x_freq_processed = torch.complex(real_part, imag_part)

        # Apply IFFT2
        spectral_output = torch.fft.irfft2(x_freq_processed, s=(x.shape[2], x.shape[3]), norm="ortho")

        spectral_weights = torch.sigmoid(spectral_output)
        out = self.outconv(z * spectral_weights) + x_d

        return out

class Attention(nn.Module):
    def __init__(self, channels, gamma = 2, b = 1):
        super(Attention, self).__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size = kernel_size, padding = (kernel_size - 1) // 2, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        v = self.avg_pool(x)
        v = self.conv(v.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        v = self.sigmoid(v)
        return x * v

class MFCAM(nn.Module):
    def __init__(self, inputchannel, midchannel=16):
        super(MFCAM, self).__init__()
        self.inplanes = midchannel
        self.deconv1_1 = DeformConv2d(inputchannel, 2*midchannel, kernel_size=1, stride=1, padding=0, bias=False)
        self.deconv3_3 = DeformConv2d(midchannel, 2*midchannel, kernel_size=3, stride=1, padding=1, bias=False)
        self.deconv5_5 = DeformConv2d(midchannel, 2*midchannel, kernel_size=5, stride=1, padding=2, bias=False)
        self.deconv7_7 = DeformConv2d(midchannel, 2*midchannel, kernel_size=7, stride=1, padding=3, bias=False)
        self.att1 = Attention(midchannel)
        self.att2 = Attention(midchannel)
        self.att3 = Attention(midchannel)
        self.att4 = Attention(midchannel)
        
        self.just = nn.Sequential(
            nn.Conv2d(5*midchannel, inputchannel, 1, bias=False),
            nn.BatchNorm2d(inputchannel),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.deconv1_1(x)
        x1_split1 = x1[:, :self.inplanes, :, :]
        x1_split2 = x1[:, self.inplanes:, :, :]
        x1_att = self.att1(x1_split1)
        
        x2 = self.deconv3_3(x1_split2)
        x2_split1 = x2[:, :self.inplanes, :, :]
        x2_split2 = x2[:, self.inplanes:, :, :]
        x2_att = self.att2(x2_split1)
        
        x3 = self.deconv5_5(x2_split2)
        x3_split1 = x3[:, :self.inplanes, :, :]
        x3_split2 = x3[:, self.inplanes:, :, :]
        x3_att = self.att3(x3_split1)
        
        x4 = self.deconv7_7(x3_split2)
        x4_split1 = x4[:, :self.inplanes, :, :]
        x4_split2 = x4[:, self.inplanes:, :, :]
        x4_att = self.att4(x4_split1)
        
        x_com = torch.cat([x1_att, x2_att, x3_att, x4_att, x4_split2], dim=1)
        
        out = self.just(x_com)
        
        return x + out

class S_encoder(nn.Module):
    def __init__(self, ):
        super(S_encoder,self).__init__()

        S_resnet = models.resnet34(pretrained=True)
        
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.encoder1 = S_resnet.layer1  #[2, 64, 256, 256]

        self.encoder1 = S_resnet.layer1  # [2, 64, 256, 256]
        self.encoder2 = S_resnet.layer2  # [2, 128, 128, 128]
        self.encoder3 = S_resnet.layer3  # [2, 256, 64, 64]
        self.encoder4 = S_resnet.layer4  # [2, 512, 32, 32]

    def forward(self, x):
        """
        S-Net Encoder forward pass.
        Input x: (B, C, 512, 512)
        Returns: h1, h2, h3, h4 (multi-scale features)
        """
        hx = spatial_fold(x, 2)
        hx = self.relu1(self.bn1(self.conv1(hx)))

        h1 = self.encoder1(hx)  # 256
        h2 = self.encoder2(h1)  # 128
        h3 = self.encoder3(h2)  # 64
        h4 = self.encoder4(h3)  # 32

        return h1, h2, h3, h4


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


class Collaborative_Attention_Matching(nn.Module):
    def __init__(self, dim, num_heads, window_size, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super(Collaborative_Attention_Matching, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, q):
        """
        x: [B, C, H, W] - provides key/value
        q: [B, C, H, W] - provides query & residual
        """
        B, C, H, W = x.shape
        Ws = self.window_size
        assert H % Ws == 0 and W % Ws == 0, "H and W must be divisible by window_size"

        def window_partition(tensor):
            # [B, C, H, W] -> [B*n_win, Ws*Ws, C]
            tensor = tensor.view(B, C, H // Ws, Ws, W // Ws, Ws)
            tensor = tensor.permute(0, 2, 4, 3, 5, 1).contiguous()
            return tensor.view(-1, Ws * Ws, C)

        def window_reverse(windows, H, W):
            B_ = int(windows.shape[0] / (H // Ws * W // Ws))
            x = windows.view(B_, H // Ws, W // Ws, Ws, Ws, C)
            x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
            return x.view(B_, C, H, W)

        # Step 1: Window partition
        x_w = window_partition(x)  # [B*n_win, N, C]
        q_w = window_partition(q)  # [B*n_win, N, C]
        Bn, N, _ = x_w.shape

        # Step 2: QKV projection
        q_proj = self.q_proj(q_w).reshape(Bn, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B*, h, N, d]
        kv = self.kv_proj(x_w).reshape(Bn, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # [B*, h, N, d]

        # Step 3: Attention calculation
        attn = (q_proj @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(Bn, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        # Step 4: Reverse spatial shape
        out = window_reverse(out, H, W)  # [B, C, H, W]

        # Step 5: Residual connection
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class CAM_Block(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=16, mlp_ratio=4.0, qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Collaborative_Attention_Matching(dim, num_heads, window_size, qkv_bias, attn_drop, drop)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, hidden_dim=int(dim * mlp_ratio), dropout=drop)

    def forward(self, x, q):
        """
        Input x: Key/Value, q: Query
        """
        B, C, H, W = x.shape
        q_ln = self.norm1(q.permute(0, 2, 3, 1))
        q_ln = q_ln.permute(0, 3, 1, 2).contiguous()

        x1 = self.attn(x, q_ln) + x  # Attention + Residual

        x2_ln = self.norm2(x1.permute(0, 2, 3, 1)).view(B, H * W, C)
        x2_mlp = self.mlp(x2_ln).view(B, H, W, C).permute(0, 3, 1, 2)

        return x1 + x2_mlp  # Final output


class Dehaze_decoder_(nn.Module):
    def __init__(self, ):
        super(Dehaze_decoder_, self).__init__()
        ############# 256-256  ##############

        self.DME1 = DynamicMutualEnhancement(256)
        self.RRDB_block1 = RRDB2(256)
        self.seg_guide1 = CAM_Block(256)
        self.recon1 = REC(256, out_c=12)

        self.DME2 = DynamicMutualEnhancement(128)
        self.RRDB_block2 = RRDB2(128)
        self.seg_guide2 = CAM_Block(128)
        self.recon2 = REC(128, out_c=12)

        self.DME3 = DynamicMutualEnhancement(64)
        self.RRDB_block3 = RRDB2(64)
        self.seg_guide3 = CAM_Block(64)
        self.recon3 = REC(64, out_c=12)

        self.SAA1 = SpectrumAwareAggregation()
        self.SAA2 = SpectrumAwareAggregation()

    def forward(self, l3, l2, l1, l0):
        h1_in = self.DME1(l1, l0)
        h1 = self.RRDB_block1(h1_in)

        h2_in = self.DME2(l2, h1)
        h2 = self.RRDB_block2(h2_in)

        h3_in = self.DME3(l3, h2)
        h3 = self.RRDB_block3(h3_in)

        return h1, h2, h3

    def refine(self, l3, l2, l1, l0, guide_d1, guide_d2, guide_d3):
        h1_in = self.DME1(l1, l0)
        h1 = self.RRDB_block1(h1_in)
        g1 = self.seg_guide1(h1, guide_d1)
        out1 = self.recon1(g1)
        out1 = spatial_unfold(out1, 2)

        h2_in = self.DME2(l2, g1)
        h2 = self.RRDB_block2(h2_in)
        g2 = self.seg_guide2(h2, guide_d2)
        out2 = self.recon2(g2)
        out2 = spatial_unfold(out2, 2)
        out1_re = F.interpolate(out1, size=out2.size()[2:], mode='bilinear', align_corners=True)
        out12 = self.SAA1(out1_re, out2)

        h3_in = self.DME3(l3, g2)
        h3 = self.RRDB_block3(h3_in)
        g3 = self.seg_guide3(h3, guide_d3)
        out3 = self.recon3(g3)
        out3 = spatial_unfold(out3, 2)
        out12_re = F.interpolate(out12, size=out3.size()[2:], mode='bilinear', align_corners=True)
        out123 = self.SAA2(out12_re, out3)

        return out1, out12, out123


class FinalHead(nn.Module):
    def __init__(self, in_ch=64 + 128 + 256, mid=128, out_ch=1):
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


class Segment_decoder_(nn.Module):
    def __init__(self, ):
        super(Segment_decoder_, self).__init__()

        self.up1 = ContextAwareAggregation(256)
        self.decoder1 = RDB_5C(256)
        self.attention1 = MFCAM(256)
        self.haze_guide1 = CAM_Block(256)
        self.out1 = REC(256, 4)

        self.up2 = ContextAwareAggregation(128)
        self.decoder2 = RDB_5C(128)
        self.attention2 = MFCAM(128)
        self.haze_guide2 = CAM_Block(128)
        self.out2 = REC(128, 4)

        self.up3 = ContextAwareAggregation(64)
        self.decoder3 = RDB_5C(64)
        self.attention3 = MFCAM(64)
        self.haze_guide3 = CAM_Block(64)
        self.out3 = REC(64, 4)
        self.aspp = ASPP(512, 512)

        self.final = FinalHead(in_ch=64 + 128 + 256, mid=128, out_ch=1)

    def refine(self, x, h1, h2, h3, h4, guide_h1, guide_h2, guide_h3):
        """
        Grad-guided segmentation refinement.
        """
        h4 = self.aspp(h4)
        up1 = self.up1(h4, h3)
        d1 = self.attention1(self.decoder1(up1))
        g1 = self.haze_guide1(d1, guide_h3)
        out1 = self.out1(g1)  # [B, 4, 64, 64]
        out1 = spatial_unfold(out1, 2)

        up2 = self.up2(g1, h2)
        d2 = self.attention2(self.decoder2(up2))
        g2 = self.haze_guide2(d2, guide_h2)
        out2 = self.out2(g2)  # [B, 4, 128, 128]
        out2 = spatial_unfold(out2, 2)

        up3 = self.up3(g2, h1)
        d3 = self.attention3(self.decoder3(up3))
        g3 = self.haze_guide3(d3, guide_h1)
        out3 = self.out3(g3)  # [B, 4, 256, 256]
        out3 = spatial_unfold(out3, 2)

        final_mask = self.final(g3, g2, g1)
        final_mask = torch.sigmoid(F.interpolate(final_mask, size=x.size()[2:], mode='bilinear', align_corners=True))

        return (torch.sigmoid(out1), torch.sigmoid(out2), torch.sigmoid(out3),
                final_mask, d1, d2, d3)


class S_network(nn.Module):
    def __init__(self):
        super(S_network, self).__init__()
        self.S_shareE = S_encoder()
        self.segment_decoder = Segment_decoder_()
        self.dehaze_decoder = Dehaze_decoder_()

    def forward(self, x):
        h1, h2, h3, h4 = self.S_shareE(x)

        # Step 1: Initialize guidance, run dehaze first
        d1, d2, d3 = self.dehaze_decoder(h1, h2, h3, h4)

        # Step 2: Dehaze intermediate features guide segmentation
        (seg_out1_refine, seg_out2_refine, seg_out3_refine,
         final_out, h1_dehaze, h2_dehaze, h3_dehaze) = self.segment_decoder.refine(
            x, h1, h2, h3, h4, d3, d2, d1)

        # Step 3: Segmentation intermediate features guide dehaze refinement
        de_out1, de_out2, de_out3 = self.dehaze_decoder.refine(
            h1, h2, h3, h4, h1_dehaze, h2_dehaze, h3_dehaze)

        return (de_out1, de_out2, de_out3, seg_out1_refine,
                seg_out2_refine, seg_out3_refine, final_out, h4, h3, h2, h1)

    def gradcam(self, x: torch.Tensor,
                target_layer: str = "S_seg.decoder3.conv5",
                use_soft_mask: bool = True):
        """
        Calculates Grad-CAM for a specific layer internally.
        Returns: cam (B, H, W tensor in [0, 1]), segout_prob (B, 1, H, W).

        Notes:
        - Performs one backward pass; must be called in eval mode.
        - Ensure target_layer is on the segout computation path.
        """
        assert not self.training, "Grad-CAM must be run in eval mode: model.eval()"
        device = x.device

        # 1) Parse target layer
        m = self
        for p in target_layer.split("."):
            m = getattr(m, p)
        target_module = m

        feats = {}
        hook = None

        def fwd_hook(mod, inp, out):
            # Retain graph for gradient retrieval
            feats["fmap"] = out
            out.retain_grad()

        # 2) Register forward hook (only once)
        hook = target_module.register_forward_hook(fwd_hook)

        try:
            # 3) Forward pass
            out = self(x)
            segout_prob = out[6]
            B, C, H, W = segout_prob.shape
            prob = segout_prob[:, 0] if C == 1 else segout_prob.mean(dim=1)

            # 4) Construct stable target scalar score (soft weighting)
            logit = torch.log(prob.clamp(1e-6, 1 - 1e-6) / (1 - prob.clamp(1e-6, 1 - 1e-6)))
            self.zero_grad(set_to_none=True)

            if use_soft_mask:
                # Soft weighting based on quantiles (per image)
                weight = []
                with torch.no_grad():
                    for b in range(B):
                        q_val = torch.quantile(prob[b], 0.80)
                        w = ((prob[b] - q_val) / (1e-6 + 1 - q_val)).clamp(0, 1).pow(2.0)
                        s = w.sum()
                        w = (w / s) if s > 0 else torch.full_like(w, 1.0 / (H * W))
                        weight.append(w)
                w = torch.stack(weight, dim=0)
                score = (logit * w).sum()
            else:
                score = logit.mean()

            # 5) Backward pass for layer gradients
            score.backward()

            assert "fmap" in feats and feats["fmap"].grad is not None, \
                f"{target_layer} received no gradient; check path to segout."

            fmap = feats["fmap"]
            grad = feats["fmap"].grad

            # 6) Grad-CAM calculation
            w = grad.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((w * fmap).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=prob.shape[-2:], mode="bilinear", align_corners=False)
            cam = cam.squeeze(1)

            # Normalize to [0, 1]
            cam_min = cam.amin(dim=(1, 2), keepdim=True)
            cam_max = cam.amax(dim=(1, 2), keepdim=True)
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-6)

            return cam.detach(), segout_prob.detach(), {"score": float(score.detach().cpu())}

        finally:
            # 7) Clean up to prevent state leak
            if hook is not None:
                hook.remove()
            feats.clear()


if __name__ == "__main__":
    model = S_network().cuda()
    x = torch.randn(2, 3, 512, 512).cuda()
    (de1, de2, de3, seg1, seg2, seg3, final, h4, h3, h2, h1) = model(x)
    print(f"Haze-free output size: {de3.size()}")
    print(f"Segmentation mask size: {final.size()}")
