import sys
sys.path.append('/nas/shared/public/wangyuan1/PQ3D/ReMamber')
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

import einops
from ReMamber.vmamba_model.vmamba import SS2D, VSSM, LayerNorm2d, Linear2d
from ReMamber.model.utils import ImageTextCorr
from ReMamber.model.customized_model import Twister, VSSBlock, VSSLayer

class ReMamber(VSSM):
    """still extract feature"""
    def __init__(self, i_layer, **kwargs):
        super().__init__(**kwargs)
        self.classifier = None
        dims = 96
        use_checkpoint = False
        norm_layer = "ln2d"
        self.text_guidencee = nn.ModuleList()
        self.local_text_fusion = nn.ModuleList()
        self.i_layer = i_layer
        
        self.multimodal_blocks = nn.ModuleList()
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )
        ssm_act_layer: nn.Module = _ACTLAYERS.get("silu", None) #_ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)


        # for i_layer in range(self.num_layers):
        layer = VSSLayer(
                dim = self.dims[i_layer],
                depth = 2,
                # drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                ssm_act_layer=ssm_act_layer,
                downsample=None,
                channel_first=self.channel_first,
                # =================
                ssm_d_state=kwargs['ssm_d_state'],
                ssm_ratio=kwargs['ssm_ratio'],
                ssm_dt_rank=kwargs['ssm_dt_rank'],
                # ssm_act_layer=kwargs['ssm_act_layer'],
                ssm_conv=kwargs['ssm_conv'],
                ssm_conv_bias=kwargs['ssm_conv_bias'],
                ssm_drop_rate=kwargs['ssm_drop_rate'],
                ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'],
                # =================
                mlp_ratio=kwargs['mlp_ratio'],
                mlp_act_layer=kwargs['mlp_act_layer'],
                mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'],
                forward_coremm='Twister',
            )
        # self.multimodal_blocks.append(layer)
        self.multimodal_blocks = layer

        guidence = nn.Sequential(
            nn.Linear(768, self.dims[i_layer]),
            nn.ReLU(),
        )
        # self.text_guidencee.append(guidence)
        self.text_guidencee = guidence

        fusing = ImageTextCorr(
            visual_dim=self.dims[i_layer],
            text_dim=768,
            hidden_dim=768,
            out_dim=self.dims[i_layer],
        )
        self.local_text_fusion = fusing

        self.projector = nn.Sequential(Linear2d(768, self.dims[i_layer]), nn.GELU())

        self.projector_final = nn.Sequential(Linear2d(self.dims[i_layer], 768), nn.GELU())

    def forward_layer(self, x, layer):
        inner = layer.blocks(x) # torch.Size([2, 96, 25, 1])
        out = layer.downsample(inner)
        return inner, out
    
    def forward(self, x_0, l_feat, l_mask, pooler_out=None): # x: query [Batch, 120, 768] --> [Batch, 768, length, 1] 

        outs = []
        x_0 = x_0.unsqueeze(-1).permute(0, 2, 1, 3)
        
        x = self.projector(x_0)
        # x, inner = self.forward_layer(x, self.layers[self.i_layer])
        _, c, h, w = x.shape   ### x

        out = x ### x
        pooling_text = torch.mean(l_feat, dim=1)
        text_guidence = self.text_guidencee(pooling_text)
        text_guidence = einops.repeat(text_guidence, "b c -> b c h w", h=h, w=w)
        # print(l_feat.shape)
        local_text = self.local_text_fusion(out, l_feat, l_mask)
        local_text = einops.rearrange(local_text, 'b h w c -> b c h w', h=h)

        # print(out.shape, text_guidence.shape, local_text.shape)
        mm_input = (out, text_guidence, local_text)
        ret = self.multimodal_blocks(mm_input, None, None)[0]
        img_feat = ret[0]
        if self.layers[self.i_layer].downsample is not None:
            x = self.layers[self.i_layer].downsample(img_feat)
        else:
            x = img_feat
        outs = self.projector_final(img_feat)
        outs = outs.squeeze(3).permute(0, 2, 1).contiguous()

        # print(outs.shape, x_0.shape)
        outs = outs + x_0.squeeze(-1).permute(0, 2, 1).contiguous()
        # print("outs: ", outs.shape)
        return outs

