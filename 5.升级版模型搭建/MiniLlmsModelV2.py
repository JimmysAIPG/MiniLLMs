# MiniLlmsModel_optimized.py

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

# 检查 flash_attn 是否可用，如果不可用则提供一个警告和回退方案的提示
try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
    print("FlashAttention is available. Using flash_attn_func.")
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    print("Warning: FlashAttention is not installed. Will use torch.nn.functional.scaled_dot_product_attention as a fallback.")
    # PyTorch 2.0+ 自带的 scaled_dot_product_attention 也是一个很好的、内存优化的替代品
    from torch.nn.functional import scaled_dot_product_attention

# 1. RMSNorm: 替代 LayerNorm
# Llama等模型中广泛使用的归一化层，比LayerNorm更简单、计算更快。
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # gamma 参数
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # 计算均方根
        # rsqrt 是平方根倒数，计算上更高效
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # (batch, seq_len, dim)
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# 2. RoPE (Rotary Positional Embedding): 替代 Sinusoidal Positional Encoding
# 一种相对位置编码，通过旋转Query和Key的嵌入向量来注入位置信息，比绝对位置编码在长序列上表现更好。
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: int = 10000, device: Optional[torch.device] = None):
        super().__init__()
        # 计算频率，inv_freq 的形状是 (dim/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        # 将 inv_freq 注册为 buffer，这样它会随模型移动到不同设备(cpu/gpu)，但不是模型参数
        self.register_buffer("inv_freq", inv_freq)

        # 预计算所有可能位置的 cos 和 sin 值
        t = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq) # (max_seq_len, dim/2)
        # freqs 包含了所有位置在所有频率上的角度
        # 将其扩展为 (max_seq_len, dim) 的形式，cos和sin交替
        emb = torch.cat((freqs, freqs), dim=-1) # (max_seq_len, dim)
        
        # freqs_cis 的形状是 (1, 1, max_seq_len, dim)
        # 预计算复数形式 e^(i*m*theta)，这里用cos和sin表示
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor):
        # x 的形状是 (bs, num_heads, seq_len, head_dim)
        seq_len = x.shape[-2]
        # 从缓存中取出对应长度的 cos 和 sin
        cos = self.cos_cached[:, :, :seq_len, ...]
        sin = self.sin_cached[:, :, :seq_len, ...]
        return cos, sin

def rotate_half(x):
    """将最后一个维度的前半部分和后半部分对调"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_emb(q, k, cos, sin):
    """应用旋转位置编码"""
    
    # 将 cos 和 sin 转换为 q 的数据类型 (例如 bfloat16)
    cos = cos.to(q.dtype)
    sin = sin.to(q.dtype)

    # q, k 的形状: (bs, num_heads, seq_len, head_dim)
    # cos, sin 的形状: (1, 1, seq_len, head_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# 3. GQA (Grouped-Query Attention) with FlashAttention: 替代 CausalSelfAttention
# GQA是MHA和MQA的折中，通过让多组Query头共享同一组Key/Value头，显著减少KV缓存，加速推理。
# FlashAttention是一种IO感知的注意力算法，避免了将巨大的注意力矩阵写入HBM，从而极大加速计算和节省显存。
class FlashGQA(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_kv_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.num_q_per_kv = num_heads // num_kv_heads # 每个KV头对应的Q头数量

        # Q, K, V 的线性投影层
        self.wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wk = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        # 输出投影层
        self.wo = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.dropout = dropout

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        # x: (batch_size, seq_len, embed_dim)
        bsz, seq_len, _ = x.shape

        # 1. 线性投影
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # 2. Reshape to (bs, seq_len, num_heads, head_dim)
        xq = xq.view(bsz, seq_len, self.num_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        # 3. 应用 RoPE
        # 首先将维度调整为 RoPE 函数期望的 (bs, num_heads, seq_len, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
        
        # 4. GQA 处理：重复 K/V 头以匹配 Q 头
        # (bs, num_kv_heads, seq_len, head_dim) -> (bs, num_heads, seq_len, head_dim)
        # 在 flash_attn_func 中，我们不需要手动重复，可以传入正确的q,k,v形状
        # flash_attn_func 的输入期望是 (batch_size, seq_len, num_heads, head_dim)
        # 所以我们需要把维度换回来
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        
        # 5. FlashAttention
        if FLASH_ATTENTION_AVAILABLE:
            # flash_attn_func可以直接处理GQA，只要传入正确的K/V头数即可
            # 我们需要将K/V头重复以匹配Q头的数量
            # [bsz, seq_len, num_kv_heads, head_dim] -> [bsz, seq_len, num_kv_heads, 1, head_dim]
            # -> [bsz, seq_len, num_kv_heads, num_q_per_kv, head_dim] -> [bsz, seq_len, num_heads, head_dim]
            keys_repeated = xk.repeat_interleave(self.num_q_per_kv, dim=2)
            values_repeated = xv.repeat_interleave(self.num_q_per_kv, dim=2)

            attn_output = flash_attn_func(
                q=xq, 
                k=keys_repeated, 
                v=values_repeated, 
                dropout_p=self.dropout if self.training else 0.0,
                causal=True
            )
        else:
            # PyTorch 2.0+ 的 scaled_dot_product_attention 作为备选
            # 它也支持 MHA, MQA, GQA. 我们需要手动调整维度和重复
            xq = xq.transpose(1, 2) # (bsz, num_heads, seq_len, head_dim)
            xk = xk.transpose(1, 2) # (bsz, num_kv_heads, seq_len, head_dim)
            xv = xv.transpose(1, 2) # (bsz, num_kv_heads, seq_len, head_dim)

            # 重复 K, V
            xk = xk.repeat_interleave(self.num_q_per_kv, dim=1)
            xv = xv.repeat_interleave(self.num_q_per_kv, dim=1)

            attn_output = scaled_dot_product_attention(
                query=xq,
                key=xk,
                value=xv,
                attn_mask=None, # causal=True 时不需要 mask
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
            # (bsz, num_heads, seq_len, head_dim) -> (bsz, seq_len, num_heads, head_dim)
            attn_output = attn_output.transpose(1, 2).contiguous()

        # 6. Reshape and Output Projection
        # (bs, seq_len, embed_dim)
        attn_output = attn_output.view(bsz, seq_len, -1)
        return self.wo(attn_output)


# 4. SwiGLU FeedForward: 替代标准 FeedForward
# SwiGLU 是一种激活函数和门控机制的结合，实验证明比ReLU/GELU效果更好。
class SwiGLUFeedForward(nn.Module):
    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.0):
        super().__init__()
        # 通常 ff_dim 是 embed_dim 的倍数，且为了SwiGLU，ff_dim常设为 2/3 * (4*embed_dim)
        # 这里我们保持 ff_dim 参数的灵活性
        self.w1 = nn.Linear(embed_dim, ff_dim, bias=False) # 门控
        self.w2 = nn.Linear(ff_dim, embed_dim, bias=False) # 输出
        self.w3 = nn.Linear(embed_dim, ff_dim, bias=False) # up projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # F.silu 是 Swish 激活函数
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# 整合所有新组件的 Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_kv_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # 使用 FlashGQA 替代 CausalSelfAttention
        self.self_attention = FlashGQA(embed_dim, num_heads, num_kv_heads, dropout)
        # 使用 SwiGLUFeedForward 替代 FeedForward
        self.feed_forward = SwiGLUFeedForward(embed_dim, ff_dim, dropout)
        
        # 使用 RMSNorm 替代 LayerNorm
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        # Pre-Norm 结构
        # 1. 注意力子层
        residual = x
        h = self.norm1(x)
        h = self.self_attention(h, freqs_cos, freqs_sin, key_padding_mask)
        x = residual + h

        # 2. 前馈网络子层
        residual = x
        h = self.norm2(x)
        h = self.feed_forward(h)
        x = residual + h
        
        return x

# 主模型：DecoderOnlyTransformer
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, num_layers: int, vocab_size: int, embed_dim: int, 
                 num_heads: int, num_kv_heads: int, ff_dim: int, max_seq_len: int, 
                 dropout: float = 0.1, padding_idx: Optional[int] = 0,
                 tie_weights: bool = True):
        super().__init__()
        
        self.padding_idx = padding_idx

        # 1. 词嵌入层 (TokenEmbedding 不再需要乘以 sqrt(d_model))
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=self.padding_idx)
        
        # 2. 删除了旧的位置编码，改为在forward中计算RoPE
        self.rotary_embedding = RotaryEmbedding(embed_dim // num_heads, max_seq_len)

        # 3. Decoder Block 堆栈
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(embed_dim, num_heads, num_kv_heads, ff_dim, dropout) for _ in range(num_layers)]
        )

        # 4. 最终的归一化层 (Llama等模型在输出前会有一个Norm)
        self.final_norm = RMSNorm(embed_dim)
        
        # 5. 输出层
        self.output_layer = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # 6. 权重绑定
        if tie_weights:
            self.output_layer.weight = self.token_embedding.weight

        # 初始化模型参数
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Llama 使用正态分布初始化
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, src: torch.Tensor):
        # src: (batch_size, seq_len)
        bsz, seq_len = src.shape
        device = src.device

        # 1. 词嵌入
        x = self.token_embedding(src) # (batch_size, seq_len, embed_dim)
        
        # 2. 获取 RoPE 的 cos 和 sin 频率
        # freqs_cis 的形状是 (1, 1, seq_len, dim)
        freqs_cos, freqs_sin = self.rotary_embedding(x)

        # 3. 通过 Decoder 块堆栈
        # 注意：padding_mask 可以在 FlashAttention 中通过特定参数处理，
        # 但在 causal attention 下，通常我们更关心序列本身的因果关系，
        # padding token 通常位于序列末尾，causal mask 会自然处理它们。
        # 如果padding在序列中间，则需要更复杂的处理。为简化，这里暂不传递。
        for block in self.decoder_blocks:
            x = block(x, freqs_cos, freqs_sin)
            
        # 4. 最终归一化和输出
        x = self.final_norm(x)
        logits = self.output_layer(x)
        
        return logits


# --- 测试 ---
if __name__ == '__main__':
    # 定义模型超参数
    num_layers = 8     # Llama-7B 约 32, 我们用一个小的
    vocab_size = 15000
    embed_dim = 512
    num_heads = 8      # Query heads
    num_kv_heads = 4   # Key/Value heads (GQA)
    ff_dim = int(2/3 * 4 * embed_dim) # SwiGLU 推荐维度
    max_seq_len = 512
    dropout_rate = 0.1
    padding_token_id = 0

    # 实例化模型
    model = DecoderOnlyTransformer(
        num_layers=num_layers,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        ff_dim=ff_dim,
        max_seq_len=max_seq_len,
        dropout=dropout_rate,
        padding_idx=padding_token_id,
        tie_weights=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print("模型结构:")
    print(model)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"\n模型总可训练参数量: {num_params:.2f}M")

    # 创建一个模拟的输入批次
    batch_size = 4
    seq_len = 256
    input_tokens = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)

    # 模型前向传播
    model.train() # 设置为训练模式以激活dropout
    print(f"\n输入Token序列形状: {input_tokens.shape}")
    logits = model(input_tokens)
    
    # 输出 logits 的形状应为 (batch_size, seq_len, vocab_size)
    print(f"输出Logits形状: {logits.shape}")
    print(f"Logits 设备: {logits.device}")