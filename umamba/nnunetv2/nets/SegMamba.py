from __future__ import annotations

import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock

from .vim_ver import Mamba


# referencing to https://github.com/ge-xing/SegMamba/blob/main/model_segmamba/segmamba.py
class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bimamba_type="v3",
            nslices=num_slices,
        )

    def forward(self, x):
        b, c = x.shape[:2]
        x_skip = x
        assert c == self.dim, f"channel mismatch: got {c}, expected {self.dim}"

        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]

        x_flat = x.reshape(b, c, n_tokens).transpose(-1, -2)   # [B, N, C]
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(b, c, *img_dims)

        return out + x_skip


class MlpChannel(nn.Module):
    def __init__(self, hidden_size, mlp_dim):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, kernel_size=1)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class GSC(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, in_channels, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channels)
        self.act = nn.ReLU(inplace=True)

        self.proj2 = nn.Conv3d(in_channels, in_channels, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channels)
        self.act2 = nn.ReLU(inplace=True)

        self.proj3 = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channels)
        self.act3 = nn.ReLU(inplace=True)

        self.proj4 = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channels)
        self.act4 = nn.ReLU(inplace=True)

    def forward(self, x):
        x_residual = x

        x1 = self.act(self.norm(self.proj(x)))
        x1 = self.act2(self.norm2(self.proj2(x1)))

        x2 = self.act3(self.norm3(self.proj3(x)))

        x = x1 + x2
        x = self.act4(self.norm4(self.proj4(x)))

        return x + x_residual


class MambaEncoder(nn.Module):
    def __init__(
        self,
        in_chans=1,
        depths=(2, 2, 2, 2),
        dims=(48, 96, 192, 384),
        out_indices=(0, 1, 2, 3),
    ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()

        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()

        # original implementation hardcodes these
        num_slices_list = [64, 32, 16, 8]

        for i in range(4):
            self.gscs.append(GSC(dims[i]))
            self.stages.append(
                nn.Sequential(*[MambaLayer(dim=dims[i], num_slices=num_slices_list[i]) for _ in range(depths[i])])
            )

        self.out_indices = out_indices
        self.mlps = nn.ModuleList()

        for i_layer in range(4):
            norm_layer = nn.InstanceNorm3d(dims[i_layer])
            self.add_module(f"norm{i_layer}", norm_layer)
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.gscs[i](x)
            x = self.stages[i](x)

            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        return self.forward_features(x)


class SegMamba(nn.Module):
    def __init__(
        self,
        in_chans=1,
        out_chans=13,
        depths=(2, 2, 2, 2),
        feat_size=(48, 96, 192, 384),
        hidden_size=768,
        norm_name="instance",
        res_block=True,
        spatial_dims=3,
    ):
        super().__init__()

        if spatial_dims != 3:
            raise NotImplementedError("This SegMamba implementation is 3D-only.")

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.feat_size = feat_size
        self.spatial_dims = spatial_dims

        self.vit = MambaEncoder(
            in_chans=in_chans,
            depths=depths,
            dims=feat_size,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_chans,
            out_channels=feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[0],
            out_channels=feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[1],
            out_channels=feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[2],
            out_channels=feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[3],
            out_channels=hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[3],
            out_channels=feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[2],
            out_channels=feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[1],
            out_channels=feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[0],
            out_channels=feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[0],
            out_channels=out_chans,
        )

    def forward(self, x_in):
        outs = self.vit(x_in)

        enc1 = self.encoder1(x_in)
        enc2 = self.encoder2(outs[0])
        enc3 = self.encoder3(outs[1])
        enc4 = self.encoder4(outs[2])
        enc_hidden = self.encoder5(outs[3])

        dec3 = self.decoder5(enc_hidden, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0)

        return self.out(out)