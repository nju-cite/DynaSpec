import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch import einsum
import os
import time


import torch
from fvcore.nn import FlopCountAnalysis

# 确保引入了必要的库，防止单独运行此片段时报错
# Make sure these are imported if you run this snippet separately
import math
from torch import nn
import torch.nn.functional as F
from einops import rearrange


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def reflect_pad(x, pad_w_right, pad_h_down):
    """
    手动实现 reflect 填充，避免使用 F.pad(mode='reflect')

    :param x: 输入张量，形状 (batch, channels, height, width)
    :param pad_w: 需要填充的宽度 (左右)
    :param pad_h: 需要填充的高度 (上下)
    :return: 经过 reflect 填充后的张量
    """
    # 获取输入的 H 和 W
    h, w = x.shape[-2], x.shape[-1]

    pad_w = pad_w_right
    pad_h = pad_h_down

    # 处理宽度填充（左右）
    if pad_w > 0:
        # left = x[..., :, 1:pad_w+1].flip(-1)  # 取左侧区域翻转
        right = x[..., :, -pad_w-1:-1].flip(-1)  # 取右侧区域翻转
        x = torch.cat([x, right], dim=-1)  # 拼接左右

    # 处理高度填充（上下）
    if pad_h > 0:
        # top = x[..., 1:pad_h+1, :].flip(-2)  # 取上方区域翻转
        bottom = x[..., -pad_h-1:-1, :].flip(-2)  # 取下方区域翻转
        x = torch.cat([x, bottom], dim=-2)  # 拼接上下

    return x

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)



class Attention_PCA(nn.Module):
    def __init__(self, dim, length, dontminus=False):
        super().__init__()
        self.dontminus = dontminus
        self.pc_proj_q = nn.Linear(dim, 1, bias=False)
        self.bias_pc_proj_q = nn.Parameter(torch.FloatTensor([1.]))
        self.pc_proj_k = nn.Linear(dim, 1, bias=False)
        self.bias_pc_proj_k = nn.Parameter(torch.FloatTensor([1.]))
        self.mlp1 = nn.Sequential(
            nn.Linear(length, 1, bias=False),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(length, length, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(length, 1, bias=False),
        )

    def forward(self, q, k, position_bias=None):
        Sigma_q = self.pc_proj_q(q) + self.bias_pc_proj_q
        Sigma_k = self.pc_proj_k(k) + self.bias_pc_proj_k
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        Sigma = einsum('b h i d, b h j d -> b h i j', Sigma_q, Sigma_k)

        if position_bias is not None:
            sim = sim + position_bias

        if self.dontminus:
            sim_norm = sim.clone()
        else:
            diag_sim = torch.diagonal(sim, dim1=-2, dim2=-1)
            sim_norm = sim - torch.diag_embed(diag_sim)
        theta = self.mlp1(sim_norm).squeeze(-1)
        theta = self.mlp2(theta).unsqueeze(-1)

        sim = sim * Sigma
        attn = sim.softmax(dim=-1) * (sim > theta)
        return attn


import torch

def generate_pooling_mask_torch(H, W, pool_H, pool_W):
    """
    生成一个池化映射 mask (torch 实现, dtype=torch.bool)。
    :param H: 原始高度
    :param W: 原始宽度
    :param pool_h, pool_w: 池化后窗口大小 (pool_h x pool_w)
    :return: nn.Parameter (dtype=torch.bool, requires_grad=False)，形状 [H//K * W//K, H * W]
    """

    KH = H // pool_H
    KW = W // pool_W

    # 生成索引矩阵 (形状 HxW)
    indices = torch.arange(H * W).reshape(H, W)

    # 计算池化后的形状
    pool_size = pool_H * pool_W

    # 创建 mask 矩阵 (dtype=torch.bool, 默认 False)
    mask = torch.zeros((pool_size, H * W), dtype=torch.bool)

    # 遍历池化后的每个位置，找到对应的池化前索引
    for i in range(pool_H):
        for j in range(pool_W):
            pool_idx = i * pool_W + j  # 计算池化后的索引
            orig_indices = indices[i * KH:(i + 1) * KH, j * KW:(j + 1) * KW].flatten()  # 获取池化前的索引
            mask[pool_idx, orig_indices] = True  # 标记对应区域

    return mask


class Attention_PCA_agent(nn.Module):
    def __init__(self, dim, length=None, before_pool_size=None, after_pool_size=None, transpose=False, dontminus=False):
        super().__init__()
        self.pc_proj_q = nn.Linear(dim, 1, bias=False)
        self.bias_pc_proj_q = nn.Parameter(torch.FloatTensor([1.]))
        self.pc_proj_k = nn.Linear(dim, 1, bias=False)
        self.bias_pc_proj_k = nn.Parameter(torch.FloatTensor([1.]))

        self.before_pool_size = before_pool_size
        self.after_pool_size = after_pool_size

        self.dontminus = dontminus
        if before_pool_size is not None:
            mask = generate_pooling_mask_torch(before_pool_size[-2],
                                               before_pool_size[-1],
                                               after_pool_size[-2],
                                               after_pool_size[-1])  #pool_size, H * W
            if transpose:
                mask = mask.transpose(0, 1)
                length_q = before_pool_size[-2] * before_pool_size[-1]
                length_k = after_pool_size[-2] * after_pool_size[-1]
            else:
                length_q = after_pool_size[-2] * after_pool_size[-1]
                length_k = before_pool_size[-2] * before_pool_size[-1]

            if not self.dontminus:
                mask = mask.unsqueeze(dim=0).unsqueeze(dim=0)
                self.register_buffer("pool_mask", mask)

        else:
            length_q = length
            length_k = length

        self.mlp1 = nn.Sequential(
            nn.Linear(length_k, 1, bias=False),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(length_q, length_q, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(length_q, 1, bias=False),
        )



    def forward(self, q, k, position_bias=None, have_thre=True):

        dontminus = self.dontminus
        Sigma_q = self.pc_proj_q(q) + self.bias_pc_proj_q
        Sigma_k = self.pc_proj_k(k) + self.bias_pc_proj_k
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        Sigma = einsum('b h i d, b h j d -> b h i j', Sigma_q, Sigma_k)

        if position_bias is not None:
            sim = sim + position_bias

        if have_thre:
            if dontminus:
                sim_norm = sim.clone()
            else:
                if self.before_pool_size is not None:
                    sim_norm = sim.clone()
                    sim_norm = sim_norm.masked_fill_(self.pool_mask, float("0"))
                    # sim_norm[self.pool_mask] = 0
                else:
                    diag_sim = torch.diagonal(sim, dim1=-2, dim2=-1)
                    sim_norm = sim - torch.diag_embed(diag_sim)
            theta = self.mlp1(sim_norm).squeeze(-1)
            theta = self.mlp2(theta).unsqueeze(-1)
            sim = sim * Sigma
            attn = sim.softmax(dim=-1) * (sim > theta)

        else:
            sim = sim * Sigma
            attn = sim.softmax(dim=-1)

        return attn

class Attention_tp(nn.Module):
    def __init__(self, dim, length):
        super().__init__()
        self.pc_proj_q = nn.Linear(dim, 1, bias=False)
        self.bias_pc_proj_q = nn.Parameter(torch.FloatTensor([1.]))
        self.pc_proj_k = nn.Linear(dim, 1, bias=False)
        self.bias_pc_proj_k = nn.Parameter(torch.FloatTensor([1.]))
        self.mlp1 = nn.Sequential(
            nn.Linear(length, 1, bias=False),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(length, length, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(length, 1, bias=False),
        )

    def forward(self, q, k):
        Sigma_q = self.pc_proj_q(q) + self.bias_pc_proj_q
        Sigma_k = self.pc_proj_k(k) + self.bias_pc_proj_k
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        Sigma = einsum('b h i d, b h j d -> b h i j', Sigma_q, Sigma_k)

        diag_sim = torch.diagonal(sim, dim1=-2, dim2=-1)
        sim_norm = sim - torch.diag_embed(diag_sim)
        theta = self.mlp1(sim_norm).squeeze(-1)
        theta = self.mlp2(theta).unsqueeze(-1)

        sim = sim * Sigma
        attn = sim.softmax(dim=-1) * (sim > theta)
        return attn

class TemSpa_Attension(nn.Module):
    def __init__(self, dim, sq_dim, dim_head=28, rank=28,  window_size=(8, 8),  input_size=[256, 256], shift=False):
        super().__init__()

        self.heads_qk = sq_dim // dim_head
        self.heads_v = dim // dim_head

        self.w_size = window_size

        self.shift = shift
        self.frame = 3
        # logit_scale = torch.log(10 * torch.ones((self.heads_qk, 1, 1)))
        # self.logit_scale_tp = nn.Parameter(logit_scale, requires_grad=True)

        # if exchange_window:
        #     window_size[-2], window_size[-1] = window_size[-1], window_size[-2]
        num_token = window_size[-2] * window_size[-1]
        self.cal_atten_sp = Attention_PCA(dim_head, num_token)
        self.cal_atten_tp = Attention_PCA(dim_head, self.frame)
        self.to_out = nn.Linear(dim, dim)

        if self.shift:
            attn_mask = self.calculate_mask(input_size[0], input_size[1], window_size)
            self.register_buffer("attn_mask", attn_mask)
        else:
            self.register_buffer("attn_mask", None)


    def vid2win_old(self, x, nh, nw, num_heads):
        x = rearrange(x, 'b t (nh h) (nw w) (d c)-> (b t nh nw) d (h w) c', d=num_heads, nh=nh, nw=nw)
        return x

    def vid2win(self, x, nh, nw, num_heads):
        x = rearrange(x, 'b  t (nh h) (nw w) (d c)-> (b t nh nw) d (h w) c', d=num_heads, nh=nh, nw=nw)
        return x

    def vid2seq(self, x, num_heads):
        x = rearrange(x, 'b t h w (d c) -> (b h w) d t c', d=num_heads)
        return x

    def calculate_mask(self, H, W, window_size):
        num_H_0, num_W_0 = H // window_size[-2], W // window_size[-1]
        shift_step = [window_size[0] // 2, window_size[1] // 2]
        attn_mask = torch.zeros(num_H_0,
                                  num_W_0,
                                  window_size[-2],
                                  window_size[-1],
                                  window_size[-2],
                                  window_size[-1],
                                  dtype=torch.bool)
        attn_mask[-1, :, :shift_step[-2], :, shift_step[-2]:, :] = True
        attn_mask[-1, :, shift_step[-2]:, :, :shift_step[-2], :] = True
        attn_mask[:, -1, :, :shift_step[-1], :, shift_step[-1]:] = True
        attn_mask[:, -1, :, shift_step[-1]:, :, :shift_step[-1]] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 ->1 1 (w1 w2) (p1 p2) (p3 p4)')

        return attn_mask

    def cal_attention(self, q, k, v, window_size):
        '''
                bthwc
        '''

        b, t, h, w, c = q.shape

        num_H = h // window_size[-2]
        num_W = w // window_size[-1]

        q_sp, k_sp = map(lambda t: self.vid2win(t, nh=num_H, nw=num_W, num_heads=self.heads_qk), (q, k))
        q_tp, k_tp = map(lambda t: self.vid2seq(t, num_heads=self.heads_qk), (q, k))
        v = self.vid2win(v, nh=num_H, nw=num_W, num_heads=self.heads_v)

        attn_sp = self.cal_atten_sp(q_sp, k_sp)
        attn_tp = self.cal_atten_tp(q_tp, k_tp)

        if self.attn_mask is not None:
            attn_sp = rearrange(attn_sp, '(b t nh nw) d hw1 hw2-> (b t) d (nh nw) hw1 hw2', b=b, nh=num_H, nw=num_W)
            attn_sp = attn_sp.masked_fill_(self.attn_mask, float("0"))
            attn_sp = rearrange(attn_sp, 'bt d nhw hw1 hw2-> (bt nhw) d hw1 hw2')

        # attn_tp = F.normalize(q_tp, dim=-1) @ F.normalize(k_tp, dim=-1).transpose(-2, -1)
        # attn_tp = attn_tp * torch.clamp(self.logit_scale_tp, max=math.log(1.0 / 0.01)).exp()
        # attn_tp = F.softmax(attn_tp, dim=-1, dtype=attn_tp.dtype)

        out_sp = einsum('b h i j, b h j d -> b h i d', attn_sp, v)
        out_sp = rearrange(out_sp,
                        '(b t nh nw) d (h w) c -> (b nh h nw w) d t c',
                        b=b,
                        t=t,
                        nh=num_H,
                        nw=num_W,
                        h=window_size[-2],
                        w=window_size[-1])
        # attn_tp = attn_tp.repeat([1, self.heads_v//self.heads_qk, 1, 1])
        out = einsum('b h i j, b h j d -> b h i d', attn_tp, out_sp)
        out = rearrange(out,
                        '(b nh h nw w) d t c -> b t (nh h) (nw w) (d c)',
                        b=b,
                        t=t,
                        nh=num_H,
                        nw=num_W,
                        h=window_size[-2],
                        w=window_size[-1])

        out = self.to_out(out)
        return out

    def forward(self, q, k, v, w_size):
        if w_size != self.w_size:
            raise ValueError(f"invalid w_size")
        out = self.cal_attention(q, k, v, w_size)
        return out


class TemSpa_Attension_Agent(nn.Module):
    def __init__(self, dim, sq_dim, dim_head=28, rank=28,  window_size=(8, 8),  input_size=[256, 256], frame=3, shift=False, agent=True):
        super().__init__()

        self.heads_qk = sq_dim // dim_head
        self.heads_v = dim // dim_head
        self.dim_head = dim_head
        self.agent = agent

        self.w_size = window_size
        num_token = window_size[-2] * window_size[-1]

        self.shift = shift
        self.frame = frame


        # self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size_h, pool_size_w))

        # logit_scale = torch.log(10 * torch.ones((self.heads_qk, 1, 1)))
        # self.logit_scale_tp = nn.Parameter(logit_scale, requires_grad=True)

        # if exchange_window:
        #     window_size[-2], window_size[-1] = window_size[-1], window_size[-2]


        if self.agent:
            self.agent_ratio = 2
            self.agent_num = num_token // self.agent_ratio // self.agent_ratio
            self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim,
                                 bias=True)
            self.an_bias = nn.Parameter(torch.zeros(self.heads_qk, self.agent_num, 8, 8))
            self.na_bias = nn.Parameter(torch.zeros(self.heads_qk, self.agent_num, 8, 8))
            self.ah_bias = nn.Parameter(torch.zeros(1, self.heads_qk, self.agent_num, window_size[0], 1))
            self.aw_bias = nn.Parameter(torch.zeros(1, self.heads_qk, self.agent_num, 1, window_size[1]))
            self.ha_bias = nn.Parameter(torch.zeros(1, self.heads_qk, window_size[0], 1, self.agent_num))
            self.wa_bias = nn.Parameter(torch.zeros(1, self.heads_qk, 1, window_size[1], self.agent_num))
            trunc_normal_(self.an_bias, std=.02)
            trunc_normal_(self.na_bias, std=.02)
            trunc_normal_(self.ah_bias, std=.02)
            trunc_normal_(self.aw_bias, std=.02)
            trunc_normal_(self.ha_bias, std=.02)
            trunc_normal_(self.wa_bias, std=.02)
            pool_size_h = int(window_size[-2] // self.agent_ratio)
            pool_size_w = int(window_size[-1] // self.agent_ratio)
            self.cal_atten_agent_linner = Attention_PCA_agent(dim_head,
                                                              before_pool_size=window_size,
                                                              after_pool_size=[pool_size_h, pool_size_w],
                                                              dontminus=True)
            self.cal_atten_agent_soft = Attention_PCA_agent(dim_head,
                                                            before_pool_size=window_size,
                                                            after_pool_size=[pool_size_h, pool_size_w],
                                                            transpose=True,
                                                            dontminus=True)
        else:
            self.cal_atten_sp = Attention_PCA(dim_head, num_token, dontminus=False)
        self.cal_atten_tp = Attention_PCA(dim_head, self.frame, dontminus=False)
        self.to_out = nn.Linear(dim, dim)

        # self.to_v_spe = nn.Linear(dim, dim, bias=False)
        # self.to_qk_spe = nn.Linear(dim, self.rank * 2, bias=False)

        # if self.shift:
        #     attn_mask = self.calculate_mask(input_size[0], input_size[1], window_size, pool=self.agent_ratio)
        #     self.register_buffer("attn_mask", attn_mask)
        # else:
        #     self.register_buffer("attn_mask", None)


    def vid2win_old(self, x, nh, nw, num_heads):
        x = rearrange(x, 'b t (nh h) (nw w) (d c)-> (b t nh nw) d (h w) c', d=num_heads, nh=nh, nw=nw)
        return x

    def vid2win(self, x, nh, nw, num_heads):
        x = rearrange(x, 'b  t (nh h) (nw w) (d c)-> (b t nh nw) d (h w) c', d=num_heads, nh=nh, nw=nw)
        return x

    def vid2seq(self, x, num_heads):
        x = rearrange(x, 'b t h w (d c) -> (b h w) d t c', d=num_heads)
        return x

    def calculate_mask(self, H, W, window_size, pool=1, transpose=False):
        num_H_0, num_W_0 = H // window_size[-2], W // window_size[-1]

        window_pool_size = [window_size[0] // pool, window_size[1] // pool]

        if transpose:
            import copy
            temp = copy.deepcopy(window_pool_size)
            window_pool_size = copy.deepcopy(window_size)
            window_size = copy.deepcopy(temp)

        shift_step = [window_size[0] // 2, window_size[1] // 2]
        shift_step_pool = [window_pool_size[0] // 2, window_pool_size[1] // 2]

        attn_mask = torch.zeros(num_H_0,
                                num_W_0,
                                window_pool_size[-2],
                                window_pool_size[-1],
                                window_size[-2],
                                window_size[-1],
                                dtype=torch.bool)
        attn_mask[-1, :, :shift_step_pool[-2], :, shift_step[-2]:, :] = True
        attn_mask[-1, :, shift_step_pool[-2]:, :, :shift_step[-2], :] = True
        attn_mask[:, -1, :, :shift_step_pool[-1], :, shift_step[-1]:] = True
        attn_mask[:, -1, :, shift_step_pool[-1]:, :, :shift_step[-1]] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 ->1 1 (w1 w2) (p1 p2) (p3 p4)')

        return attn_mask

    def cal_attention(self, q, k, v, x, window_size):
        '''
                bthwc
        '''

        b, t, h, w, c = x.shape

        h_win = window_size[-2]
        w_win = window_size[-1]

        num_H = h // window_size[-2]
        num_W = w // window_size[-1]

        q_sp, k_sp = map(lambda t: self.vid2win(t, nh=num_H, nw=num_W, num_heads=self.heads_qk), (q, k)) # n d hw c
        q_tp, k_tp = map(lambda t: self.vid2seq(t, num_heads=self.heads_qk), (q, k))
        v = self.vid2win(v, nh=num_H, nw=num_W, num_heads=self.heads_v)
        # x = self.vid2win(x, nh=num_H, nw=num_W, num_heads=1) # n 1 hw c

        # attn_sp = self.cal_atten_sp(q_sp, k_sp)
        attn_tp = self.cal_atten_tp(q_tp, k_tp)

        if self.agent:
            b_sp, d_sp, win2, c_sp = q_sp.shape
            agent_tokens = F.avg_pool2d(input=q_sp.reshape(b_sp*d_sp, h_win, w_win, c_sp).permute(0, 3, 1, 2),
                                        kernel_size=(self.agent_ratio, self.agent_ratio)).reshape(b_sp, d_sp, c_sp, -1).permute(0, 1, 3, 2)
            # agent_tokens = self.pool(q_sp.reshape(b_sp*d_sp, h_win, w_win, c_sp).permute(0, 3, 1, 2)).reshape(b_sp, d_sp, c_sp, -1).permute(0, 1, 3, 2)
            position_bias1 = nn.functional.interpolate(self.an_bias, size=self.w_size, mode='bilinear')
            position_bias1 = position_bias1.reshape(1, self.heads_qk, self.agent_num, -1).repeat(b, 1, 1, 1)
            position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, self.heads_qk, self.agent_num, -1).repeat(b, 1, 1, 1)
            position_bias = position_bias1 + position_bias2
            agent_attn_linner = self.cal_atten_agent_linner(agent_tokens, k_sp, position_bias)


            agent_bias1 = nn.functional.interpolate(self.na_bias, size=self.w_size, mode='bilinear')
            agent_bias1 = agent_bias1.reshape(1, self.heads_qk, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
            agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, self.heads_qk, -1, self.agent_num).repeat(b, 1, 1, 1)
            agent_bias = agent_bias1 + agent_bias2
            agent_attn_soft = self.cal_atten_agent_soft(q_sp, agent_tokens, agent_bias)

            # if self.attn_mask is not None:
            #     agent_attn_linner = rearrange(agent_attn_linner, '(b t nh nw) d hw1 hw2-> (b t) d (nh nw) hw1 hw2', b=b, nh=num_H, nw=num_W)
            #     agent_attn_linner = agent_attn_linner.masked_fill_(self.attn_mask, float("0"))
            #     agent_attn_linner = rearrange(agent_attn_linner, 'bt d nhw hw1 hw2-> (bt nhw) d hw1 hw2')
            #     agent_attn_soft = rearrange(agent_attn_soft, '(b t nh nw) d hw1 hw2-> (b t) d (nh nw) hw1 hw2', b=b, nh=num_H, nw=num_W)
            #     agent_attn_soft = agent_attn_soft.masked_fill_(self.attn_mask.transpose(-1, -2), float("0"))
            #     agent_attn_soft = rearrange(agent_attn_soft, 'bt d nhw hw1 hw2-> (bt nhw) d hw1 hw2')

            agent_v = einsum('b h i j, b h j d -> b h i d', agent_attn_linner, v)
            out = einsum('b h i j, b h j d -> b h i d', agent_attn_soft, agent_v) # n dv hw dim_head

            b_sp = out.shape[0]

            out = rearrange(out,
                            '(b t nh nw) d (h w) c -> (b t) (d c) (nh h) (nw w)',
                            b=b,
                            t=t,
                            nh=num_H,
                            nw=num_W,
                            h=window_size[-2],
                            w=window_size[-1])
            out = self.dwc(out) + x.reshape(b*t,h,w,c).permute(0,3,1,2)
            out = out.permute(0,2,3,1).reshape(b,t,h,w,c)
            out = self.vid2seq(out, num_heads=self.heads_v)
            # out = out.transpose(1, 2).reshape(b_sp, h_win, w_win, c).permute(0, 3, 1, 2)
            # out = x.reshape(b_sp, h_win, w_win, c).permute(0, 3, 1, 2) + self.dwc(out)
            # out = out.permute(0, 2, 3, 1).reshape(b_sp, h_win*w_win, self.heads_v, self.dim_head).transpose(1, 2)
        else:
            attn_sp = self.cal_atten_sp(q_sp, k_sp)
            # if self.attn_mask is not None:
            #     attn_sp = rearrange(attn_sp, '(b t nh nw) d hw1 hw2-> (b t) d (nh nw) hw1 hw2', b=b, nh=num_H, nw=num_W)
            #     attn_sp = attn_sp.masked_fill_(self.attn_mask, float("0"))
            #     attn_sp = rearrange(attn_sp, 'bt d nhw hw1 hw2-> (bt nhw) d hw1 hw2')
            # attn_tp = F.normalize(q_tp, dim=-1) @ F.normalize(k_tp, dim=-1).transpose(-2, -1)
            # attn_tp = attn_tp * torch.clamp(self.logit_scale_tp, max=math.log(1.0 / 0.01)).exp()
            # attn_tp = F.softmax(attn_tp, dim=-1, dtype=attn_tp.dtype)
            out = einsum('b h i j, b h j d -> b h i d', attn_sp, v)
            out = rearrange(out,
                            '(b t nh nw) d (h w) c -> (b nh h nw w) d t c',
                            b=b,
                            t=t,
                            nh=num_H,
                            nw=num_W,
                            h=window_size[-2],
                            w=window_size[-1])

        # out = rearrange(out,
        #                 '(b t nh nw) d (h w) c -> (b nh h nw w) d t c',
        #                 b=b,
        #                 t=t,
        #                 nh=num_H,
        #                 nw=num_W,
        #                 h=window_size[-2],
        #                 w=window_size[-1])
        # attn_tp = attn_tp.repeat([1, self.heads_v//self.heads_qk, 1, 1])
        out = einsum('b h i j, b h j d -> b h i d', attn_tp, out)
        out = rearrange(out,
                        '(b nh h nw w) d t c -> b t (nh h) (nw w) (d c)',
                        b=b,
                        t=t,
                        nh=num_H,
                        nw=num_W,
                        h=window_size[-2],
                        w=window_size[-1])

        out = self.to_out(out)
        return out

    def forward(self, q, k, v, x, w_size):
        if w_size != self.w_size:
            raise ValueError(f"invalid w_size")
        out = self.cal_attention(q, k, v, x, w_size)
        return out

class CDPA(nn.Module):
    '''
        spatiotemporal feature focus attention
    '''
    def __init__(self, dim, window_size=(8, 8), dim_head=28, sq_dim=None, input_size=[256, 256], frame=3, shift=False, agent=True):
        super().__init__()

        self.branch_num = 2
        self.shift = shift

        if sq_dim is None:
            self.rank = dim
        else:
            self.rank = sq_dim

        self.window_size = window_size.copy()
        self.w_size_exchange = window_size.copy()
        self.w_size_exchange[-2], self.w_size_exchange[-1] = self.w_size_exchange[-1], self.w_size_exchange[-2]
        # self.shift_step = [self.window_size[0]//2, self.window_size[1]//2]

        self.to_v = nn.Linear(dim, dim * self.branch_num, bias=False)
        self.to_qk = nn.Linear(dim, self.rank * 2 * self.branch_num, bias=False)

        self.attns = nn.ModuleList([
            TemSpa_Attension_Agent(dim=dim,
                                   sq_dim=sq_dim,
                                   dim_head=dim_head,
                                   rank=self.rank,
                                   window_size=self.w_size_exchange if (i % 2) else self.window_size,
                                   input_size=input_size,
                                   frame=frame,
                                   shift=shift,
                                   agent=agent)
            for i in range(self.branch_num)])

        if self.branch_num == 2:
            self.proj_out = nn.Linear(dim * self.branch_num, dim)


    def shift_feat(self, q, k, v, x=None, window_size=[8,8]):
        '''
        bthwc
        '''
        if self.shift:
            shift_step = [window_size[0]//2, window_size[1]//2]
            q = q.roll(shifts=(-shift_step[-2], -shift_step[-1]), dims=(-3, -2))
            k = k.roll(shifts=(-shift_step[-2], -shift_step[-1]), dims=(-3, -2))
            v = v.roll(shifts=(-shift_step[-2], -shift_step[-1]), dims=(-3, -2))
            if x is not None:
                x = x.roll(shifts=(-shift_step[-2], -shift_step[-1]), dims=(-3, -2))
        if x is not None:
            return (q, k, v, x)
        else:
            return (q, k, v)

    def shift_back(self, x, window_size):
        '''
        bthwc
        '''
        if self.shift:
            shift_step = [window_size[0] // 2, window_size[1] // 2]
            x = x.roll(shifts=(shift_step[-2], shift_step[-1]), dims=(-3, -2))
        return x

    def forward(self, x, f, h, w):
        '''
           x:bthwc
        '''
        b, t, h, w, c = x.shape
        w_size = self.window_size
        w_size_exchange = self.w_size_exchange


        q, k = self.to_qk(x).chunk(2, dim=-1)
        v = self.to_v(x)

        if self.branch_num == 2:
            q, q_1 = q.chunk(2, dim=-1)
            k, k_1 = k.chunk(2, dim=-1)
            v, v_1 = v.chunk(2, dim=-1)

        qkvx_feat = self.shift_feat(q, k, v, x, w_size)
        (q, k, v, x) = qkvx_feat
        # x_inp = rearrange(x, 'b t (h b0) (w b1) c -> (b t h w) (b0 b1) c', b0=w_size[0], b1=w_size[1])

        out = self.attns[0](q, k, v, x, w_size)
        out = self.shift_back(out, w_size)

        if self.branch_num == 2:
            (q_1, k_1, v_1) = self.shift_feat(q_1, k_1, v_1, x=None, window_size=w_size_exchange)
            out_1 = self.attns[1](q_1, k_1, v_1, x, w_size_exchange)
            out_1 = self.shift_back(out_1, w_size_exchange)
            out = torch.concat([out, out_1], dim=-1)
            out = self.proj_out(out)

        return out


class MPMLP_2d(nn.Module):
    def __init__(self, dim, multi=4):
        super(MPMLP_2d, self).__init__()

        self.multi = multi
        self.pwconv1 = nn.Sequential(
            nn.Conv2d(dim, dim * multi, 1, groups=dim, bias=False),
            GELU(),
        )
        self.groupconv = nn.Sequential(
            nn.Conv2d(dim * multi, dim * multi, 1, groups=multi, bias=False),
            GELU(),
        )
        self.pwconv2 = nn.Conv2d(dim * multi, dim, 1, groups=dim, bias=False)

    def forward(self, x):
        '''
        x bthwc
        '''
        b, t, h, w, c = x.shape
        x = x.reshape(b*t, h, w, c)
        x = self.pwconv1(x.permute(0, 3, 1, 2))
        x = rearrange(x, 'b (c m) h w -> b (m c) h w', m=self.multi)
        x = self.groupconv(x)
        x = rearrange(x, 'b (m c) h w -> b (c m) h w', m=self.multi)
        x = self.pwconv2(x).permute(0, 2, 3, 1)
        x = x.reshape(b, t, h, w, c)
        return x


class MDFFN(nn.Module):
    def __init__(self, dim, multi=4):
        super(MDFFN, self).__init__()

        self.multi = multi
        self.half_multi = self.multi // 2
        self.pwconv1 = nn.Sequential(
            nn.Conv3d(dim, dim * multi, 1, groups=dim, bias=False),
            GELU(),
        )
        self.spconv = nn.Sequential(
            nn.Conv3d(dim * self.half_multi, dim * self.half_multi, kernel_size=(1, 3, 3),
                      padding=(0, 1, 1), groups=self.half_multi, bias=False),
            GELU(),
        )

        self.tempconv = nn.Sequential(
            nn.Conv3d(dim * self.half_multi, dim * self.half_multi, kernel_size=(3, 1, 1),
                      padding=(1, 0, 0), groups=self.half_multi, bias=False),
            GELU(),
        )

        self.groupconv = nn.Sequential(
            nn.Conv3d(dim * self.multi, dim * self.multi, 1, groups=self.multi, bias=False),
            GELU(),
        )

        self.pwconv2 = nn.Conv3d(dim * multi, dim, 1, groups=dim, bias=False)

    def forward(self, x):
        '''
        x bthwc  from fvcore.nn import FlopCountAnalysis
        '''
        b, t, h, w, c = x.shape
        x = self.pwconv1(x.permute(0, 4, 1, 2, 3))
        x = rearrange(x, 'b (c m) t h w -> b (m c) t h w', m=self.multi)
        x_s, x_t = x.chunk(2, dim=1)
        x_s = self.spconv(x_s)
        x_t = self.tempconv(x_t)
        x = self.groupconv(torch.cat([x_s, x_t], dim=1))
        x = rearrange(x, 'b (m c) t h w -> b (c m) t h w', m=self.multi)
        x = self.pwconv2(x).permute(0, 2, 3, 4, 1)
        return x

class CDPB(nn.Module):
    '''
     SpatioTemporal Feature Propagate Block
    '''
    def __init__(self, dim, sq_dim,  dim_head=28, window_size=(8, 8), mult=4, input_size=[256, 256], shift=False, frame=3, agent=True):
        super().__init__()

        # self.pos_emb = nn.Conv2d(dim, dim, 5, 1, 2, bias=False, groups=dim)
        self.pos_emb = nn.Conv3d(dim, dim, (3, 5, 5), 1, (1, 2, 2), bias=False, groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.cdpa = CDPA(dim=dim,
                         window_size=window_size,
                         dim_head=dim_head,
                         sq_dim=sq_dim,
                         input_size=input_size,
                         frame=frame,
                         shift=shift,
                         agent=agent
                         )

        self.norm2 = nn.LayerNorm(dim)
        self.mdffn = MDFFN(dim=dim, multi=mult)
        self.frame = frame

    def forward(self, x):
        nf, c, h, w = x.shape
        f = self.frame
        n = nf // f
        x = x.reshape(n, f, c, h, w).permute(0, 2, 1, 3, 4)
        x = x + self.pos_emb(x)
        x = x.permute(0, 2, 3, 4, 1)  # bthwc
        x_ = self.norm1(x)
        x = self.cdpa(x_, f, h, w) + x
        x_ = self.norm2(x)
        x = self.mdffn(x_) + x
        x = x.permute(0, 1, 4, 2, 3).reshape(nf, c, h, w)
        return x


class DSR(nn.Module):
    '''
        dynamic spectral reconstruction
    '''
    def __init__(self, in_dim=56, out_dim=28, dim_head=28, numblock=[2, 4, 4], window_size=[8, 32], frame_num=3):
        super(DSR, self).__init__()
        print("window size:", window_size)
        self.mult = 2
        self.numblock = numblock
        self.window_size = window_size
        # self.window_size2 = window_size
        # self.window_size2 = [x // 2 for x in window_size]
        self.window_size2 = [8, 8]
        self.stages = len(numblock)
        self.dim_head = dim_head
        self.shuffle_conv = nn.Parameter(torch.cat([torch.ones(dim_head, 1, 1, 1), torch.zeros(dim_head, 1, 1, 1)], dim=1))
        self.conv_in = nn.Conv2d(in_dim, dim_head, 3, 1, 1, bias=False)
        self.down1 = self._make_layer(block=CDPB,
                                      dim=dim_head,
                                      sq_dim=dim_head,
                                      dim_head=dim_head,
                                      window_size=self.window_size,
                                      mult=self.mult,
                                      layer_num=self.numblock[0],
                                      input_size=[256, 256],
                                      shift=False,
                                      frame=frame_num,
                                      agent=True)
        # self.down1 = FAB(dim=dim_head, sq_dim=dim_head, dim_head=dim_head, mult=4)
        self.downsample1 = nn.Conv2d(dim_head, 2*dim_head, 4, 2, 1, bias=False)
        self.down2 = self._make_layer(block=CDPB,
                                      dim=2*dim_head,
                                      sq_dim=dim_head,
                                      dim_head=dim_head,
                                      window_size=self.window_size2,
                                      mult=self.mult,
                                      layer_num=self.numblock[1],
                                      input_size=[128, 128],
                                      shift=False,
                                      frame=frame_num,
                                      agent=False)
        # self.down2 = FAB(dim=2*dim_head, sq_dim=dim_head, dim_head=dim_head, mult=4)
        self.downsample2 = nn.Conv2d(dim_head*2, dim_head*4, 4, 2, 1, bias=False)

        self.bottleneck_local = self._make_layer(block=CDPB,
                                                 dim=2 * dim_head,
                                                 sq_dim=dim_head,
                                                 dim_head=dim_head,
                                                 window_size=self.window_size2,
                                                 mult=self.mult,
                                                 layer_num=self.numblock[2],
                                                 input_size=[64, 64],
                                                 shift=False,
                                                 frame=frame_num,
                                                 agent=False)
        # self.bottleneck_local = FAB(dim=dim_head*2, sq_dim=dim_head, dim_head=dim_head, mult=4)

        self.bottleneck_swin = self._make_layer(block=CDPB,
                                                dim=2 * dim_head,
                                                sq_dim=dim_head,
                                                dim_head=dim_head,
                                                window_size=self.window_size2,
                                                mult=self.mult,
                                                layer_num=self.numblock[2],
                                                input_size=[64, 64],
                                                shift=True,
                                                frame=frame_num,
                                                agent=False)
        # self.bottleneck_swin = FAB(dim=dim_head*2, sq_dim=dim_head, dim_head=dim_head, mult=4, shift=True)


        self.upsample2 = nn.ConvTranspose2d(dim_head*4, dim_head*2, 2, 2)
        self.fusion2 = nn.Conv2d(dim_head*4, dim_head*2, 1, 1, 0, bias=False)

        self.up2 = self._make_layer(block=CDPB,
                                    dim=2 * dim_head,
                                    sq_dim=dim_head,
                                    dim_head=dim_head,
                                    window_size=self.window_size2,
                                    mult=self.mult,
                                    layer_num=self.numblock[1],
                                    input_size=[128, 128],
                                    shift=True,
                                    frame=frame_num,
                                    agent=False)
        # self.up2 = FAB(dim=dim_head*2, sq_dim=dim_head, dim_head=dim_head, mult=4, shift=True)
        self.upsample1 = nn.ConvTranspose2d(dim_head*2, dim_head, 2, 2)
        self.fusion1 = nn.Conv2d(dim_head*2, dim_head, 1, 1, 0, bias=False)

        self.up1 = self._make_layer(block=CDPB,
                                    dim=dim_head,
                                    sq_dim=dim_head,
                                    dim_head=dim_head,
                                    window_size=self.window_size,
                                    mult=self.mult,
                                    layer_num=self.numblock[0],
                                    input_size=[256, 256],
                                    shift=True,
                                    frame=frame_num,
                                    agent=True)
        # self.up1 = FAB(dim=dim_head, sq_dim=dim_head, dim_head=dim_head, mult=4, shift=True)
        self.conv_out = nn.Conv2d(dim_head, out_dim, 3, 1, 1, bias=False)

        self.apply(self._init_weights)

    def _make_layer(self, block, layer_num=1, **kwarg):
        layers = []
        for i in range(layer_num):
            layers.append(block(**kwarg))
        return nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        b, c, h_inp, w_inp = x.shape
        hw_b = math.lcm(self.window_size[-2], self.window_size[-1])
        # hb, wb = 16, 16
        pad_h = (hw_b - h_inp % hw_b) % hw_b
        pad_w = (hw_b - w_inp % hw_b) % hw_b
        if pad_w != 0 or pad_h != 0:
            # x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')  #right down
            x = reflect_pad(x, pad_w_right=pad_w, pad_h_down=pad_h)

        x = rearrange(x, 'b (n c) h w -> b (c n) h w', n=2)
        x_in = F.conv2d(x, self.shuffle_conv, groups=self.dim_head)

        x = self.conv_in(x)
        x1 = self.down1(x)
        x = self.downsample1(x1)
        x2 = self.down2(x)
        x = self.downsample2(x2)

        x_local = self.bottleneck_local(x[:, :self.dim_head*2, :, :])
        x_swin = self.bottleneck_swin(x[:, self.dim_head*2:, :, :] + x_local)
        x = torch.cat([x_local, x_swin], dim=1)

        x = self.upsample2(x)
        x = x2 + self.fusion2(torch.cat([x, x2], dim=1))
        x = self.up2(x)
        x = self.upsample1(x)
        x = x1 + self.fusion1(torch.cat([x, x1], dim=1))
        x = self.up1(x)
        out = self.conv_out(x) + x_in

        return out[:, :, :h_inp, :w_inp]



class MGDP(nn.Module):
    '''
        mask-guide degradation perception
    '''
    def __init__(self, in_dim=56, nC=28):
        super().__init__()
        self.nC = nC
        self.norm_n = nn.LayerNorm(self.nC*2)
        self.norm_mask = nn.LayerNorm(self.nC*2)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_dim, self.nC, 1, 1, 0, bias=False),
            GELU(),
        )
        self.weight = nn.Sequential(
            nn.Conv2d(self.nC*2, self.nC, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Conv2d(self.nC, self.nC, 1, 1, 0, bias=False),
        )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight.data, std=.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, Phi=None, Phi_compre=None):

        x = self.norm_n(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.fusion(x)
        mask = self.norm_mask(torch.cat([Phi, Phi_compre], dim=1).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        weight = self.weight(mask)
        return self.out(x * weight)

class PGDSRT_sd(torch.nn.Module):
    def __init__(self,  bands, size, step, frame, numblock=[2, 4, 4]):
        super(PGDSRT_sd, self).__init__()
        self.nC = bands
        self.size = size
        self.step = step
        self.frame = frame
        # self.conv = nn.Conv2d(self.nC * 2, self.nC, 1, 1, 0)

        self.mgdp = MGDP(in_dim=self.nC*2, nC=self.nC)
        self.dsr = DSR(in_dim=self.nC*2, out_dim=self.nC, dim_head=self.nC, numblock=numblock, frame_num=self.frame)


        filename = os.path.abspath(__file__)
        net_name = os.path.splitext(os.path.split(filename)[1])[0]
        self.net_name = net_name
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(net_name)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')


    def reverse_sd(self, x, len_shift=2):
        for i in range(self.nC):
            x[:, i, :, :] = torch.roll(x[:, i, :, :], shifts=(-1) * len_shift * i, dims=2)
        return x[:, :, :, :self.size]

    def shift_sd(self, x, len_shift=2):
        x = F.pad(x, [0, self.nC*len_shift-len_shift, 0, 0], mode='constant', value=0)
        for i in range(self.nC):
            x[:, i, :, :] = torch.roll(x[:, i, :, :], shifts=len_shift * i, dims=2)
        return x

    # def shift_3d_back_sd(self, inputs, step=1):  # input [bs,256,310]  output [bs, 28, 256, 256]
    #     [bs, nC, row, col] = inputs.shape
    #     output = torch.zeros(bs, nC, row, col - (nC - 1) * step).cuda().float()
    #     for i in range(nC):
    #         output[:, i, :, :] = inputs[:, i, :, step * i:step * i + col - (nC - 1) * step]
    #     return output


    def forward(self, g, input_mask=None):
        '''
           input g:nf h w  phi nf c h w  PhiPhiT nf h w
        '''
        Phi, PhiPhiT = input_mask
        Phi_shift = Phi.clone()
        Phi = self.reverse_sd(Phi, self.step)
        PhiPhiT = PhiPhiT.unsqueeze(dim=1)
        g = g.unsqueeze(dim=1)
        Phi_compressive = torch.sum(Phi_shift, dim=1, keepdim=True)
        Phi_compressive = Phi_compressive / self.nC * 2
        Phi_compressive = self.reverse_sd(Phi_compressive.repeat(1, self.nC, 1, 1), len_shift=self.step)
        g_normal = g / self.nC * 2
        temp_g = g_normal.repeat(1, self.nC, 1, 1)
        f0 = self.reverse_sd(temp_g, len_shift=self.step)

        # f = self.conv(torch.cat([f0, Phi], dim=1))
        f = self.mgdp(torch.cat([f0, Phi], dim=1), Phi, Phi_compressive)
        out = self.dsr(torch.cat([f, f0], dim=1))

        return out

class PGDSRT_dd(torch.nn.Module):
    def __init__(self,  bands, size, step, frame, numblock=[2, 4, 4]):
        super(PGDSRT_dd, self).__init__()
        self.nC = bands
        self.size = size
        self.step = step
        self.frame = frame
        # self.conv = nn.Conv2d(self.nC * 2, self.nC, 1, 1, 0)

        self.mgdp = MGDP(in_dim=self.nC*2, nC=self.nC)
        self.dsr = DSR(in_dim=self.nC*2, out_dim=self.nC, dim_head=self.nC, numblock=numblock, frame_num=self.frame)


        filename = os.path.abspath(__file__)
        net_name = os.path.splitext(os.path.split(filename)[1])[0]
        self.net_name = net_name
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(net_name)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')


    def reverse_dd(self, x, len_shift=2):
        len_shift = 0
        for i in range(self.nC):
            x[:, i, :, :] = torch.roll(x[:, i, :, :], shifts=(-1) * len_shift * i, dims=2)
        return x[:, :, :, :self.size]

    def shift_dd(self, x, len_shift=2):
        len_shift = 0
        x = F.pad(x, [0, self.nC*len_shift-len_shift, 0, 0], mode='constant', value=0)
        for i in range(self.nC):
            x[:, i, :, :] = torch.roll(x[:, i, :, :], shifts=len_shift * i, dims=2)
        return x

    # def mul_PhiTg_dd(self, Phi_shift, g, len_shift=2):
    #     temp_1 = g.repeat(1, Phi_shift.shape[1], 1, 1).cuda()
    #     PhiTg = temp_1 * Phi_shift
    #     PhiTg = self.reverse_dd(PhiTg, len_shift=len_shift)
    #     return PhiTg
    #
    # def mul_Phif_dd(self, Phi_shift, f, len_shift=2):
    #     f_shift = self.shift_dd(f, len_shift=len_shift)
    #     Phif = Phi_shift * f_shift
    #     Phif = torch.sum(Phif, 1)
    #     return Phif.unsqueeze(1)

    def forward(self, g, input_mask=None):
        '''
           input g:nf h w  phi nf c h w  PhiPhiT nf h w
        '''
        Phi, PhiPhiT = input_mask
        PhiPhiT = PhiPhiT.unsqueeze(dim=1)
        g = g.unsqueeze(dim=1)
        Phi_shift = self.shift_dd(Phi, len_shift=self.step)
        Phi_compressive = torch.sum(Phi_shift, dim=1, keepdim=True)
        Phi_compressive = Phi_compressive / self.nC * 2
        Phi_compressive = self.reverse_dd(Phi_compressive.repeat(1, self.nC, 1, 1), len_shift=self.step)
        g_normal = g / self.nC * 2
        temp_g = g_normal.repeat(1, self.nC, 1, 1)
        f0 = self.reverse_dd(temp_g, len_shift=self.step)

        # f = self.conv(torch.cat([f0, Phi], dim=1))
        f = self.mgdp(torch.cat([f0, Phi], dim=1), Phi, Phi_compressive)
        out = self.dsr(torch.cat([f, f0], dim=1))

        return out


def analyze_full_model_composition_cpu(H=128, W=128, C=28, Frame=3):
    """
    为了避免 CUDA OOM，我们在 CPU 上进行 FLOPs 统计。
    注意：为了加快速度，我默认将 H, W 调小到了 128。
    如果你需要严格的 256 输入下的数值，可以将 H, W 改回 256，但 CPU 可能会跑得稍慢一点。
    """
    device = torch.device('cpu')  # 强制使用 CPU
    print(f"{'=' * 30} 全网络模块级消耗统计 (运行在 {device}) {'=' * 30}")

    # 1. 实例化完整模型
    # 确保模型里没有写死的 .cuda()，如果有，PyTorch 通常会自动忽略或需要手动去除
    try:
        full_model = PGDSRT_dd(bands=C, size=H, step=2, frame=Frame, numblock=[2, 4, 4])
        full_model = full_model.to(device)
        full_model.eval()
    except Exception as e:
        print(f"模型初始化失败，请检查代码中是否有写死的 .cuda(): {e}")
        return

    # 2. 构造输入 (全部在 CPU 上)
    g = torch.randn(1 * Frame, H, W, device=device)
    Phi = torch.randn(1 * Frame, C, H, W, device=device)
    PhiPhiT = torch.randn(1 * Frame, H, W, device=device)

    inputs = (g, (Phi, PhiPhiT))

    # 3. 运行分析
    print(f"正在分析 (Input: {H}x{W})... 请耐心等待...")

    # 捕获可能的 Trace 错误
    try:
        flops_analyzer = FlopCountAnalysis(full_model, inputs)
        flops_analyzer.unsupported_ops_warnings(False)
        flops_dict = flops_analyzer.by_module()
    except RuntimeError as e:
        print(f"分析过程中出错: {e}")
        return

    # 4. 统计逻辑
    stats = {
        "CDPA": {"flops": 0, "params": 0, "count": 0},
        "MDFFN": {"flops": 0, "params": 0, "count": 0},
        "MGDP": {"flops": 0, "params": 0, "count": 0}
    }

    for name, module in full_model.named_modules():
        if name == "": continue

        # 获取 FLOPs
        current_flops = flops_dict.get(name, 0)

        # 统计
        if isinstance(module, CDPA):
            stats["CDPA"]["flops"] += current_flops
            stats["CDPA"]["params"] += sum(p.numel() for p in module.parameters())
            stats["CDPA"]["count"] += 1
        elif isinstance(module, MDFFN):
            stats["MDFFN"]["flops"] += current_flops
            stats["MDFFN"]["params"] += sum(p.numel() for p in module.parameters())
            stats["MDFFN"]["count"] += 1
        elif isinstance(module, MGDP):
            stats["MGDP"]["flops"] += current_flops
            stats["MGDP"]["params"] += sum(p.numel() for p in module.parameters())
            stats["MGDP"]["count"] += 1

    # 计算总量
    total_flops = flops_analyzer.total()
    total_params = sum(p.numel() for p in full_model.parameters())

    # 5. 打印结果
    print(f"\n全网总计 (Total Model) @ Input {H}x{W}:")
    print(f"  GMacs : {total_flops / (1024 ** 3):.4f} G")
    print(f"  Params: {total_params / 1e6:.4f} M")

    print("-" * 75)
    print(f"{'Module':<10} | {'Count':<5} | {'GMacs (Sum)':<15} | {'Params (Sum)':<15} | {'Ratio'}")
    print("-" * 75)

    for key in ["CDPA", "MDFFN", "MGDP"]:
        f_val = stats[key]["flops"]
        p_val = stats[key]["params"]
        cnt = stats[key]["count"]

        gmacs = f_val / (1024 ** 3)
        params_m = p_val / 1e6
        ratio = (f_val / total_flops) * 100 if total_flops > 0 else 0

        print(f"{key:<10} | {cnt:<5} | {gmacs:<15.4f} | {params_m:<15.4f} | {ratio:.2f}%")
    print("-" * 75)


if __name__ == "__main__":
    # 建议先用 128 测试通过，如果需要精确的 256 数值再改回 256
    analyze_full_model_composition_cpu(H=256, W=256, C=28, Frame=3)