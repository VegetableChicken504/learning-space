import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        out_dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.q_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.v_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, v):
        B, N, _ = q.shape
        C = self.out_dim
        k = v
        NK = k.size(1)

        q = (
            self.q_map(q)
            .view(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k_map(k)
            .view(B, NK, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v_map(v)
            .view(B, NK, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        weights = attn
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, weights


# 准备数据
input_a = torch.randn(16, 64, 192)  # 输入序列A，大小为(batch_size, seq_len_a, input_dim_a)
input_b = torch.randn(16, 192, 192)  # 输入序列B，大小为(batch_size, seq_len_b, input_dim_b)
# 定义模型
input_dim_a = input_a.shape[-1]

hidden_dim = 64
cross_attention = CrossAttention(input_dim_a, 64, 8,False,None,0.0,0.0)

# 前向传播
x1, weghts1 = cross_attention(input_a, input_b)
print("Adjusted output A:\n",x1.shape)
