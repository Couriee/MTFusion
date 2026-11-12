import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import math
from einops import rearrange
import numpy as np


def validate_curvature(curvature):
    if not math.isfinite(curvature) or curvature <= 0:
        raise ValueError("Curvature must be finite and strictly positive.")
    return curvature


def safe_project(x, c=1.0):
    """Keep each channel vector inside the Poincare ball."""
    margin = max(1e-5, torch.finfo(x.dtype).eps)
    max_norm = (1.0 - margin) / torch.as_tensor(c, device=x.device, dtype=x.dtype).sqrt()
    norm = x.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(x.dtype).eps)
    return x * (max_norm / norm).clamp(max=1.0)


def exp_map(tangent, c):
    radius = c.sqrt() * tangent.norm(dim=-1, keepdim=True)
    eps = torch.finfo(tangent.dtype).eps
    scale = torch.where(radius < eps, 1.0, radius.tanh() / radius.clamp_min(eps))
    return safe_project(tangent * scale, c)


def log_map(point, c):
    radius = c.sqrt() * point.norm(dim=-1, keepdim=True)
    eps = torch.finfo(point.dtype).eps
    scale = torch.where(radius < eps, 1.0, radius.atanh() / radius.clamp_min(eps))
    return point * scale


def mobius_translation(image, anchor, c):
    """Translate by the negative anchor using equation (2)."""
    difference = image - anchor
    image_slack = 1 - c * image.square().sum(dim=-1, keepdim=True)
    anchor_slack = 1 - c * anchor.square().sum(dim=-1, keepdim=True)
    squared_difference = c * difference.square().sum(dim=-1, keepdim=True)
    # Algebraically equivalent to the dot-product form. This denominator
    # avoids subtracting nearly equal terms near the boundary of the ball.
    numerator = image_slack * difference + squared_difference * image
    denominator = image_slack * anchor_slack + squared_difference
    return safe_project(numerator / denominator, c)


def poincare_distance(u, v, c=1.0):
    """Poincare distance in its numerically stable asinh form."""
    c = torch.as_tensor(c, device=u.device, dtype=u.dtype)
    sqrt_c = c.sqrt()
    u_slack = 1 - c * u.square().sum(dim=-1)
    v_slack = 1 - c * v.square().sum(dim=-1)
    delta = (u - v).norm(dim=-1)
    return (2 / sqrt_c) * torch.asinh(sqrt_c * delta / (u_slack * v_slack).sqrt())


def hyperbolic_norm(x, c=1.0):
    sqrt_c = torch.as_tensor(c, device=x.device, dtype=x.dtype).sqrt()
    return (2 / sqrt_c) * torch.atanh(sqrt_c * x.norm(dim=-1))


def poincare_loss(translated, anchor, c, tau=0.5):
    """One weighted distance per image and scale, following equations (3)-(4)."""
    image_norm = hyperbolic_norm(translated, c).mean(dim=1)
    anchor_norm = hyperbolic_norm(anchor, c).squeeze(1)
    weight = torch.exp(-(image_norm - anchor_norm).abs() / tau)
    return weight * poincare_distance(translated, anchor, c).mean(dim=1)


def shared_text_anchor(condition, guidance_levels):
    # All scales use the same fixed curvature. Keep the four legacy parameter
    # slots so existing model and optimizer checkpoints retain their layout.
    dtype = torch.float64 if condition.dtype == torch.float64 else torch.float32
    log_curvature = torch.stack([level.log_c.to(dtype=dtype) for level in guidance_levels]).mean()
    curvature = log_curvature.exp()
    anchor_width = guidance_levels[0].image_proj.out_features
    projections = [
        level.project_text(condition).to(dtype=dtype).reshape(condition.shape[0], -1, anchor_width).mean(dim=1)
        for level in guidance_levels
    ]
    anchor = exp_map(torch.stack(projections).mean(dim=0).unsqueeze(1), curvature)
    return anchor, curvature


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, channels, curvature=1.0):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.LeakyReLU(),
            nn.Linear(in_channels * 2, channels * 2),
        )
        self.log_c = nn.Parameter(torch.tensor(math.log(validate_curvature(curvature))), requires_grad=False)
        self.image_proj = nn.Linear(channels, channels)
        self.text_proj = nn.Linear(channels, channels)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # The caller resolves curvature from checkpoint metadata or an explicit
        # experiment setting. Old checkpoints carry unused log_c values here.
        key = prefix + "log_c"
        if key in state_dict:
            state_dict[key] = self.log_c.detach()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def project_text(self, condition):
        gamma, _ = self.MLP(condition).chunk(2, dim=-1)
        return self.text_proj(gamma)

    def forward(self, x, anchor, curvature, *, measure_distance=True):
        batch, channels, height, width = x.shape
        patches = x.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        image = exp_map(self.image_proj(patches).to(dtype=anchor.dtype), curvature)
        # Embed the shared anchor in each scale's channel subspace with zeros.
        # Padding both arguments to the widest scale gives the same geometry.
        anchor = F.pad(anchor, (0, channels - anchor.shape[-1]))
        translated = mobius_translation(image, anchor, curvature)
        distance = poincare_loss(translated, anchor, curvature) if measure_distance else None
        enhanced = log_map(translated, curvature).reshape(batch, height, width, channels)
        return enhanced.permute(0, 3, 1, 2).to(dtype=x.dtype), distance


def print_euclidean_distances(features):
    """打印特征间的欧式距离矩阵"""
    dist_matrix = torch.cdist(features, features, p=2)
    print("距离矩阵形状:", dist_matrix.shape)
    print("样本0到其他样本的距离:", dist_matrix[0])


def custom_attention(image_feature, text_feature,patch_size):

    b, c, w, h = image_feature.shape
    # patch_size = 8

    num_patches = (w // patch_size) * (h // patch_size)
    # print(image_feature.shape)
    # 图像特征的 patch embedding
    image_patches = image_feature.view(b, c, w // patch_size, patch_size, h // patch_size, patch_size)
    image_patches = image_patches.permute(0, 2, 4, 1, 3, 5).contiguous()
    image_patches = image_patches.view(b, c, num_patches, patch_size * patch_size )

    return image_patches





def min_max_normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val)

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            # out = F.normalize(out)
            out = F.relu(out, inplace=True)
            # out = self.dropout(out)
        return out

class PatchFlattener1(nn.Module):
    def __init__(self, patch_size):
        super(PatchFlattener1, self).__init__()
        self.patch_size = patch_size

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        assert height % self.patch_size == 0 and width % self.patch_size == 0, "Invalid patch size"
        num_patches = (height // self.patch_size) * (width // self.patch_size)
        patch_height = height // self.patch_size
        patch_width = width // self.patch_size
        # Rearrange patches into columns
        x = x.view(batch_size, channels, patch_height, self.patch_size, patch_width, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(batch_size, num_patches, -1)
        return x

#

class TEDFusion(nn.Module):
    def __init__(self, model_clip, inp_A_channels=3, inp_B_channels=3, out_channels=3,
                 dim=48, num_blocks=[2, 2, 2, 2],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 curvature=1.0,
                 ):

        super(TEDFusion, self).__init__()

        self.model_clip = model_clip
        self.model_clip.eval()

        self.encoder_A = Encoder_A(inp_channels=inp_A_channels, dim=dim, num_blocks=num_blocks, heads=heads,
                                   ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

        self.encoder_B = Encoder_B(inp_channels=inp_B_channels, dim=dim, num_blocks=num_blocks, heads=heads,
                                   ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

        self.cross_attention = Cross_attention(dim * 2 ** 3)
        self.attention_spatial = Attention_spatial(dim * 2 ** 3)

        self.feature_fusion_4 = Fusion_Embed(embed_dim=dim * 2 ** 3)
        self.prompt_guidance_4 = FeatureWiseAffine(256, dim * 2 ** 3, curvature)
        self.decoder_level4 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.feature_fusion_3 = Fusion_Embed(embed_dim = dim * 2 ** 2)
        self.prompt_guidance_3 = FeatureWiseAffine(256, dim * 2 ** 2, curvature)
        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.feature_fusion_2 = Fusion_Embed(embed_dim = dim * 2 ** 1)
        self.prompt_guidance_2 = FeatureWiseAffine(256, dim * 2 ** 1, curvature)
        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.feature_fusion_1 = Fusion_Embed(embed_dim = dim)
        self.prompt_guidance_1 = FeatureWiseAffine(256, dim, curvature)
        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.up2_1_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level1 = nn.Conv2d(432, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(96, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=96, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.output1 = nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output3 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output4 = nn.Conv2d(16, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.conv48to384 = nn.Conv2d(48, 384, kernel_size=3, stride=1, padding=1, bias=bias)

        self.decoder = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=bias)

        self.isss = 0

        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)



        inp_channels = 1
        out_channels = 1
        dim = 64
        num_blocks = [4, 4]
        heads = [8, 8, 8]
        ffn_expansion_factor = 2
        bias = False
        LayerNorm_type = 'WithBias'

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        stride = 1
        output_nc = 1

        self.attr_proj1 = nn.Linear(2, 64)
        self.attr_proj2 = nn.Linear(2, 64)
        self.attr_proj3 = nn.Linear(2, 64)
        self.attr_proj4 = nn.Linear(2, 64)

        self.attr_proj5 = nn.Linear(1, 256)
        # 文本语义映射网络
        self.text_proj = nn.Linear(154, 256)
        # 动态门控融合器
        self.fusion_gate = nn.Sequential(
            nn.Linear(128, 64),
            nn.Sigmoid()
        )

        self.text_proj21 = nn.Linear(512, 32)
        self.text_proj22 = nn.Linear(512, 32)

        self.is_training = 1
        self.drop_prob = 0.3
        # self.drop_prob = 1
        # self.dummy_text = nn.Parameter(torch.randn(1, 64))

    def forward(self, inp_img_A, inp_img_B, text1,text2,feature):
        b = inp_img_A.shape[0]
        # text1 = text1.float()
        # text2 = text2.float()
        # text = text.mean(dim=0)

        # feature = torch.relu(self.fc1(feature))
        # feature = self.fc2(feature)
        # feature = torch.sigmoid(feature)

        if feature.ndim == 1 and b == 1:
            feature = feature.unsqueeze(0)
        if feature.shape != (b, 8):
            raise ValueError("Expected per-image attributes [{},8], got {}".format(
                b, tuple(feature.shape)
            ))

        f1,f2,f3,f4 = feature.chunk(4, dim=1)



        # print(f1.shape)
        # 属性特征映射
        h_attr1 = self.attr_proj1(f1).expand(b, -1)
        h_attr2 = self.attr_proj2(f2).expand(b, -1)
        h_attr3 = self.attr_proj3(f3).expand(b, -1)
        h_attr4 = self.attr_proj4(f4).expand(b, -1)

        h_text1 = self.get_text_feature(text1.expand(b, -1)).to(inp_img_A.dtype)
        h_text2 = self.get_text_feature(text2.expand(b, -1)).to(inp_img_A.dtype)

        h_text1 = self.text_proj21(h_text1)
        h_text2 = self.text_proj22(h_text2)
        h_text = torch.cat([h_text1,h_text2],dim=1)


        if self.training and self.is_training:
            replace_text = torch.rand_like(h_text[:, :1]) < self.drop_prob
        else:
            replace_text = torch.full_like(h_text[:, :1], not self.is_training, dtype=torch.bool)

        fused_parts = []
        for attribute in (h_attr1, h_attr2, h_attr3, h_attr4):
            condition = torch.where(replace_text, attribute, h_text)
            gate = self.fusion_gate(torch.cat([attribute, condition], dim=1))
            fused_parts.append(torch.lerp(condition, attribute, gate))

        fused = torch.cat(fused_parts, dim=1)
        fusedd = torch.stack(fused_parts).sum(dim=0)
        anchor, curvature = shared_text_anchor(fused, (
            self.prompt_guidance_1, self.prompt_guidance_2,
            self.prompt_guidance_3, self.prompt_guidance_4,
        ))
        self.isss += 1

        x4_A, x3_A, x2_A, x1_A = self.encoder_A(inp_img_A) #4:[384,12,12] 3:[192,24,24] 2:[96,48,48] 1:[48,96,96]
        x4_B, x3_B, x2_B, x1_B = self.encoder_B(inp_img_B)

        x4_A = self.attention_spatial(x4_A)
        x4_B = self.attention_spatial(x4_B)



        x4_A, x4_B = self.cross_attention(x4_A, x4_B)

        x4 = self.feature_fusion_4(x4_A, x4_B)
        # x4 = self.attention_spatial(x4)

        x4, distance4 = self.prompt_guidance_4(x4, anchor, curvature)
        inp4 = x4
        outd4 = self.decoder_level4(inp4)

        inp3 = self.up4_3(outd4)
        inp3, distance3 = self.prompt_guidance_3(inp3, anchor, curvature)
        x3 = self.feature_fusion_3(x3_A, x3_B)
        inp31 = torch.cat([inp3, x3], 1)
        inp31 = self.reduce_chan_level3(inp31)
        outd31 = self.decoder_level3(inp31)

        inp2 = self.up3_2(outd31)
        inp2, distance2 = self.prompt_guidance_2(inp2, anchor, curvature)
        x2 = self.feature_fusion_2(x2_A, x2_B)
        inp21 = torch.cat([x2, self.up3_2(x3)], 1)
        inp22 = torch.cat([inp21,x2,inp2], 1)
        inp22 = self.reduce_chan_level2(inp22)
        outd22 = self.decoder_level2(inp22)

        inp1 = self.up2_1(outd22)
        inp1, distance1 = self.prompt_guidance_1(inp1, anchor, curvature)
        x1 = self.feature_fusion_1(x1_A, x1_B)
        inp11 = torch.cat([x1, self.up2_1(x2)], 1)

        inp12 = torch.cat([x1, inp11, self.up2_1_2(inp21)], 1)
        inp13 = torch.cat([x1, inp11, inp12, inp1], 1)

        inp13 = self.reduce_chan_level1(inp13)

        outd1 = self.decoder_level1(inp13)

        outd1 = self.refinement(outd1)



        outd1 = self.output1(outd1)
        outd1 = self.output2(outd1)
        outd1 = self.output3(outd1)
        outd1 = self.output4(outd1)



        distance_all = torch.stack((distance1, distance2, distance3, distance4)).mean()

        return [outd1,h_text,fusedd,distance_all]

    @torch.no_grad()
    def get_text_feature(self, text):
        text_feature = self.model_clip.encode_text(text)
        return text_feature

class TEDFusion1(nn.Module):
    # All four TIAM levels are active, but remain weak to preserve metrics.
    tiam_strength = 0.002

    def __init__(self, model_clip, inp_A_channels=3, inp_B_channels=3, out_channels=3,
                 dim=48, num_blocks=[2, 2, 2, 2],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 curvature=1.0,
                 ):

        super(TEDFusion1, self).__init__()
        # self.dummy_text = nn.Parameter(torch.randn(1, 64))
        self.model_clip = model_clip
        self.model_clip.eval()

        self.encoder_A = Encoder_A(inp_channels=inp_A_channels, dim=dim, num_blocks=num_blocks, heads=heads,
                                   ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

        self.encoder_B = Encoder_B(inp_channels=inp_B_channels, dim=dim, num_blocks=num_blocks, heads=heads,
                                   ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

        self.cross_attention = Cross_attention(dim * 2 ** 3)
        self.attention_spatial = Attention_spatial(dim * 2 ** 3)

        self.feature_fusion_4 = Fusion_Embed(embed_dim=dim * 2 ** 3)
        self.prompt_guidance_4 = FeatureWiseAffine(256, dim * 2 ** 3, curvature)
        self.decoder_level4 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.feature_fusion_3 = Fusion_Embed(embed_dim = dim * 2 ** 2)
        self.prompt_guidance_3 = FeatureWiseAffine(256, dim * 2 ** 2, curvature)
        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.feature_fusion_2 = Fusion_Embed(embed_dim = dim * 2 ** 1)
        self.prompt_guidance_2 = FeatureWiseAffine(256, dim * 2 ** 1, curvature)
        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.feature_fusion_1 = Fusion_Embed(embed_dim = dim)
        self.prompt_guidance_1 = FeatureWiseAffine(256, dim, curvature)
        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.up2_1_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level1 = nn.Conv2d(432, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        for guidance in (
                self.prompt_guidance_4,
                self.prompt_guidance_3,
                self.prompt_guidance_2,
                self.prompt_guidance_1):
            guidance.reuse_registered_text_proj = True
        # self.decoder_level1 = nn.Sequential(*[
        #     TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
        #                      bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(96, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=96, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.output1 = nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output3 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output4 = nn.Conv2d(16, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.decoder = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=bias)

        self.isss = 0

        # self.fc1 = nn.Linear(8, 16)
        # self.fc2 = nn.Linear(16, 4)

        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

        # self.MLP = nn.Sequential(
        #     nn.Linear(4, 4 * 2),
        #     nn.LeakyReLU(),
        #     nn.Linear(4 * 2, 4)
        # )


        inp_channels = 1
        out_channels = 1
        dim = 64
        num_blocks = [4, 4]
        heads = [8, 8, 8]
        ffn_expansion_factor = 2
        bias = False
        LayerNorm_type = 'WithBias'

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.attr_proj1 = nn.Linear(2, 64)
        self.attr_proj2 = nn.Linear(2, 64)
        self.attr_proj3 = nn.Linear(2, 64)
        self.attr_proj4 = nn.Linear(2, 64)
        self.attr_proj5 = nn.Linear(1, 256)

        # 文本语义映射网络
        self.text_proj = nn.Linear(154, 256)
        # 动态门控融合器
        self.fusion_gate = nn.Sequential(
            nn.Linear(128, 64),
            nn.Sigmoid()
        )
        # self.fusion_gate = nn.Sequential(
        #     nn.Linear(512, 256),
        #     nn.Sigmoid()
        # )

        self.text_proj21 = nn.Linear(512, 32)
        self.text_proj22 = nn.Linear(512, 32)
        # self.text_proj21 = nn.Linear(512, 128)
        # self.text_proj22 = nn.Linear(512, 128)
        # self.conv48to384 = nn.Conv2d(48, 384, kernel_size=3, stride=1, padding=1, bias=bias)
        self.is_training = 0
        self.temp_storage = []
        self.all_samples = []
        self.sample_names = []

    def _apply_weak_tiam(self, x, anchor, curvature, gate_control, guidance_module):
        guided, _ = guidance_module(x, anchor, curvature, measure_distance=False)
        strength = self.tiam_strength * (0.5 + gate_control)
        return x + strength * (guided - x)

    def forward(self, inp_img_A, inp_img_B,feature):
        b = inp_img_A.shape[0]
        if feature.ndim == 1 and b == 1:
            feature = feature.unsqueeze(0)
        if feature.shape != (b, 8):
            raise ValueError("Expected per-image attributes [{},8], got {}".format(
                b, tuple(feature.shape)
            ))

        # ATGFM inference: attributes replace the unavailable text branch.
        f1, f2, f3, f4 = feature.chunk(4, dim=1)
        h_attr1 = self.attr_proj1(f1)
        h_attr2 = self.attr_proj2(f2)
        h_attr3 = self.attr_proj3(f3)
        h_attr4 = self.attr_proj4(f4)

        gates = [
            self.fusion_gate(torch.cat([h_attr1, h_attr1], dim=1)),
            self.fusion_gate(torch.cat([h_attr2, h_attr2], dim=1)),
            self.fusion_gate(torch.cat([h_attr3, h_attr3], dim=1)),
            self.fusion_gate(torch.cat([h_attr4, h_attr4], dim=1)),
        ]
        fused = torch.cat([h_attr1, h_attr2, h_attr3, h_attr4], dim=1)
        anchor, curvature = shared_text_anchor(fused, (
            self.prompt_guidance_1, self.prompt_guidance_2,
            self.prompt_guidance_3, self.prompt_guidance_4,
        ))
        gate_control = torch.cat(gates, dim=1).mean(
            dim=1, keepdim=True
        ).view(b, 1, 1, 1)

        self.isss = self.isss + 1

        x4_A, x3_A, x2_A, x1_A = self.encoder_A(inp_img_A) #4:[384,12,12] 3:[192,24,24] 2:[96,48,48] 1:[48,96,96]
        x4_B, x3_B, x2_B, x1_B = self.encoder_B(inp_img_B)

        x4_A = self.attention_spatial(x4_A)
        x4_B = self.attention_spatial(x4_B)
        x4_A, x4_B = self.cross_attention(x4_A, x4_B)
        x4 = self.feature_fusion_4(x4_A, x4_B)
        x4 = self._apply_weak_tiam(
            x4, anchor, curvature, gate_control, self.prompt_guidance_4
        )
        inp4 = x4
        outd4 = self.decoder_level4(inp4)

        inp3 = self.up4_3(outd4)
        inp3 = self._apply_weak_tiam(
            inp3, anchor, curvature, gate_control, self.prompt_guidance_3
        )
        x3 = self.feature_fusion_3(x3_A, x3_B)
        inp31 = torch.cat([inp3, x3], 1)
        inp31 = self.reduce_chan_level3(inp31)
        outd31 = self.decoder_level3(inp31)

        inp2 = self.up3_2(outd31)
        inp2 = self._apply_weak_tiam(
            inp2, anchor, curvature, gate_control, self.prompt_guidance_2
        )
        x2 = self.feature_fusion_2(x2_A, x2_B)
        inp21 = torch.cat([x2, self.up3_2(x3)], 1)
        inp22 = torch.cat([inp21,x2,inp2], 1)
        inp22 = self.reduce_chan_level2(inp22)
        outd22 = self.decoder_level2(inp22)

        inp1 = self.up2_1(outd22)
        inp1 = self._apply_weak_tiam(
            inp1, anchor, curvature, gate_control, self.prompt_guidance_1
        )
        x1 = self.feature_fusion_1(x1_A, x1_B)
        inp11 = torch.cat([x1, self.up2_1(x2)], 1)

        inp12 = torch.cat([x1, inp11, self.up2_1_2(inp21)], 1)
        inp13 = torch.cat([x1, inp11, inp12, inp1], 1)

        inp13 = self.reduce_chan_level1(inp13)

        outd1 = self.decoder_level1(inp13)

        outd1 = self.refinement(outd1)



        outd1 = self.output1(outd1)
        outd1 = self.output2(outd1)
        outd1 = self.output3(outd1)
        outd1 = self.output4(outd1)
        return outd1

    @torch.no_grad()
    def get_text_feature(self, text):
        text_feature = self.model_clip.encode_text(text)
        return text_feature


class Cross_attention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()
        self.n_head = n_head
        self.norm_A = nn.GroupNorm(norm_groups, in_channel)
        self.norm_B = nn.GroupNorm(norm_groups, in_channel)
        self.qkv_A = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_A = nn.Conv2d(in_channel, in_channel, 1)

        self.qkv_B = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_B = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, x_A, x_B):
        batch, channel, height, width = x_A.shape

        n_head = self.n_head
        head_dim = channel // n_head

        x_A = self.norm_A(x_A)
        qkv_A = self.qkv_A(x_A).view(batch, n_head, head_dim * 3, height, width)
        query_A, key_A, value_A = qkv_A.chunk(3, dim=2)

        x_B = self.norm_B(x_B)
        qkv_B = self.qkv_B(x_B).view(batch, n_head, head_dim * 3, height, width)
        query_B, key_B, value_B = qkv_B.chunk(3, dim=2)

        attn_A = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query_B, key_A
        ).contiguous() / math.sqrt(channel)
        attn_A = attn_A.view(batch, n_head, height, width, -1)
        attn_A = torch.softmax(attn_A, -1)
        attn_A = attn_A.view(batch, n_head, height, width, height, width)

        out_A = torch.einsum("bnhwyx, bncyx -> bnchw", attn_A, value_A).contiguous()
        out_A = self.out_A(out_A.view(batch, channel, height, width))
        out_A = out_A + x_A

        attn_B = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query_A, key_B
        ).contiguous() / math.sqrt(channel)
        attn_B = attn_B.view(batch, n_head, height, width, -1)
        attn_B = torch.softmax(attn_B, -1)
        attn_B = attn_B.view(batch, n_head, height, width, height, width)

        out_B = torch.einsum("bnhwyx, bncyx -> bnchw", attn_B, value_B).contiguous()
        out_B = self.out_B(out_B.view(batch, channel, height, width))
        out_B = out_B + x_B

        return out_A, out_B

class Attention_spatial(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head
        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)
        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input











class Encoder_A(nn.Module):
    def __init__(self, inp_channels=3, dim=32, num_blocks=[2, 3, 3, 4], heads=[1, 2, 4, 8], ffn_expansion_factor=2.66, bias=False,
                 LayerNorm_type='WithBias'):
        super(Encoder_A, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.encoder_level4 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

    def forward(self, inp_img_A):
        inp_enc_level1_A = self.patch_embed(inp_img_A)
        out_enc_level1_A = self.encoder_level1(inp_enc_level1_A)

        inp_enc_level2_A = self.down1_2(out_enc_level1_A)
        out_enc_level2_A = self.encoder_level2(inp_enc_level2_A)

        inp_enc_level3_A = self.down2_3(out_enc_level2_A)
        out_enc_level3_A = self.encoder_level3(inp_enc_level3_A)

        inp_enc_level4_A = self.down3_4(out_enc_level3_A)
        out_enc_level4_A = self.encoder_level4(inp_enc_level4_A)

        return out_enc_level4_A, out_enc_level3_A, out_enc_level2_A, out_enc_level1_A


class Encoder_B(nn.Module):
    def __init__(self, inp_channels=1, dim=32, num_blocks=[2, 3, 3, 4], heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias'):
        super(Encoder_B, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.encoder_level4 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

    def forward(self, inp_img_B):
        inp_enc_level1_B = self.patch_embed(inp_img_B)
        out_enc_level1_B = self.encoder_level1(inp_enc_level1_B)

        inp_enc_level2_B = self.down1_2(out_enc_level1_B)
        out_enc_level2_B = self.encoder_level2(inp_enc_level2_B)

        inp_enc_level3_B = self.down2_3(out_enc_level2_B)
        out_enc_level3_B = self.encoder_level3(inp_enc_level3_B)

        inp_enc_level4_B = self.down3_4(out_enc_level3_B)
        out_enc_level4_B = self.encoder_level4(inp_enc_level4_B)

        return out_enc_level4_B, out_enc_level3_B, out_enc_level2_B, out_enc_level1_B

class Fusion_Embed(nn.Module):
    def __init__(self, embed_dim, bias=False):
        super(Fusion_Embed, self).__init__()

        self.fusion_proj = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x_A, x_B):
        x = torch.concat([x_A, x_B], dim=1)
        x = self.fusion_proj(x)
        return x

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


# Compatibility aliases for scripts using the Text_IF class names.
Text_IF = TEDFusion
Text_IF1 = TEDFusion1
