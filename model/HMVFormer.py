
import torch
import torch.nn as nn
from functools import partial
from einops import rearrange
from timm.models.layers import DropPath

from common.opt import opts

from model.QMNN import QMN
from model.Spatial_Encoder import First_view_Spatial_features, Spatial_features
from model.TemTemporal_Encoder import TemTemporal__features

opt = opts().parse()
device = torch.device("cuda")


#######################################################################################################################
class hmvformer(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        embed_dim = embed_dim_ratio * num_joints
        out_dim = num_joints * 3  #### output dimension is num_joints * 3

        ##Spatial_features
        self.SF1 = First_view_Spatial_features(num_frame, num_joints, in_chans, embed_dim_ratio, depth,
                                               num_heads, mlp_ratio, qkv_bias, qk_scale,
                                               drop_rate, attn_drop_rate, drop_path_rate, norm_layer)
        self.SF2 = Spatial_features(num_frame, num_joints, in_chans, embed_dim_ratio, depth,
                                    num_heads, mlp_ratio, qkv_bias, qk_scale,
                                    drop_rate, attn_drop_rate, drop_path_rate, norm_layer)
        self.SF3 = Spatial_features(num_frame, num_joints, in_chans, embed_dim_ratio, depth,
                                    num_heads, mlp_ratio, qkv_bias, qk_scale,
                                    drop_rate, attn_drop_rate, drop_path_rate, norm_layer)
        self.SF4 = Spatial_features(num_frame, num_joints, in_chans, embed_dim_ratio, depth,
                                    num_heads, mlp_ratio, qkv_bias, qk_scale,
                                    drop_rate, attn_drop_rate, drop_path_rate, norm_layer)

        ## MVF
        self.view_pos_embed = nn.Parameter(torch.zeros(1, 4, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=0.)

        self.conv_real = nn.Sequential(
            nn.BatchNorm2d(4, momentum=0.1),
            nn.Conv2d(4, 1, kernel_size=opt.mvf_kernel, stride=1, padding=int(opt.mvf_kernel // 2), bias=False),
            nn.ReLU(inplace=True),
        )
        self.conv_norm_real = nn.LayerNorm(embed_dim)


        self.conv_img = nn.Sequential(
            nn.BatchNorm2d(4, momentum=0.1),
            nn.Conv2d(4, 1, kernel_size=opt.mvf_kernel, stride=1, padding=int(opt.mvf_kernel // 2), bias=False),
            nn.ReLU(inplace=True),
        )
        self.conv_norm_img = nn.LayerNorm(embed_dim)

        # Time Serial
        self.TF = TemTemporal__features(num_frame, num_joints, in_chans, embed_dim_ratio, depth,
                                        num_heads, mlp_ratio, qkv_bias, qk_scale,
                                        drop_rate, attn_drop_rate, drop_path_rate, norm_layer)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

        self.hop_w0 = nn.Parameter(torch.ones(17, 17))
        self.hop_w1 = nn.Parameter(torch.ones(17, 17))
        self.hop_w2 = nn.Parameter(torch.ones(17, 17))
        self.hop_w3 = nn.Parameter(torch.ones(17, 17))
        self.hop_w4 = nn.Parameter(torch.ones(17, 17))

        self.hop_global = nn.Parameter(torch.ones(17, 17))

        self.linear_hop = nn.Linear(8, 2)


        self.qmn = QMN(num_frame)

    def forward(self, x, hops):
        b, f, v, j, c = x.shape

        ###############################local feature##################################
        x_hop1 = torch.einsum('bfvjc, jj -> bfvjc', torch.einsum('bjj, bfvjc -> bfvjc', hops[:, 0], x), self.hop_w1)
        x_hop2 = torch.einsum('bfvjc, jj -> bfvjc', torch.einsum('bjj, bfvjc -> bfvjc', hops[:, 1], x), self.hop_w2)
        # x_hop3 = torch.einsum('bfvjc, jj -> bfvjc', torch.einsum('bjj, bfvjc -> bfvjc', hops[:,2], x), self.hop_w3)
        # x_hop4 = torch.einsum('bfvjc, jj -> bfvjc', torch.einsum('bjj, bfvjc -> bfvjc', hops[:,3], x), self.hop_w4)

        ################################global feature##################################
        x_hop_global = x.unsqueeze(3).repeat(1, 1, 1, 17, 1, 1)
        x_hop_global = x_hop_global - x_hop_global.permute(0, 1, 2, 4, 3, 5)
        x_hop_global = torch.sum(x_hop_global ** 2, dim=-1)
        hop_global = x_hop_global / torch.sum(x_hop_global, dim=-1).unsqueeze(-1)
        x_hop_global = torch.einsum('bfvjc, jj -> bfvjc', torch.einsum('bfvjj, bfvjc -> bfvjc', hop_global, x),
                                    self.hop_global)

        # x = self.linear_hop(torch.cat([x, x_hop1, x_hop2, x_hop3, x_hop4, x_hop_global], dim=-1))
        x = self.linear_hop(torch.cat([x, x_hop1, x_hop2, x_hop_global], dim=-1))

        x1 = x[:, :, 0]
        x2 = x[:, :, 1]
        x3 = x[:, :, 2]
        x4 = x[:, :, 3]

        x1 = x1.permute(0, 3, 1, 2)
        x2 = x2.permute(0, 3, 1, 2)
        x3 = x3.permute(0, 3, 1, 2)
        x4 = x4.permute(0, 3, 1, 2)

        x1, MSA1, MSA2, MSA3, MSA4 = self.SF1(x1)
        x2, MSA1, MSA2, MSA3, MSA4 = self.SF2(x2, MSA1, MSA2, MSA3, MSA4)
        x3, MSA1, MSA2, MSA3, MSA4 = self.SF3(x3, MSA1, MSA2, MSA3, MSA4)
        x4, MSA1, MSA2, MSA3, MSA4 = self.SF4(x4, MSA1, MSA2, MSA3, MSA4)

        ################## Decomposition of real and imaginary parts##################################
        x_real_img = self.qmn([x1, x2, x3, x4])

        x_img = torch.cat((x_real_img[0][1].unsqueeze(1), \
                x_real_img[1][1].unsqueeze(1), \
                x_real_img[2][1].unsqueeze(1), \
                x_real_img[3][1].unsqueeze(1)), dim=1) + self.view_pos_embed
        
        x_real = torch.cat((x_real_img[0][0].unsqueeze(1), \
                x_real_img[1][0].unsqueeze(1), \
                x_real_img[2][0].unsqueeze(1), \
                x_real_img[3][0].unsqueeze(1)), dim=1) + self.view_pos_embed
        
        x_img, x_real = self.pos_drop(x_img), self.pos_drop(x_real)

        x_real = self.conv_norm_real(self.conv_real(x_real).squeeze(1) + torch.sum(x_real, dim=1))
        x_img = self.conv_norm_img(self.conv_img(x_img).squeeze(1) + torch.sum(x_img, dim=1))

        x = self.TF(x_real, x_img)
        x = self.head(x)
        x = x.view(b, opt.frames, j, -1)

        return x