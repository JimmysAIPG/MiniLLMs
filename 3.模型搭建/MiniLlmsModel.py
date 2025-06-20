from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, padding_idx: Optional[int] = None):
        super(TokenEmbedding, self).__init__()
        # nn.Embedding: PyTorch提供的标准嵌入层。
        # vocab_size: 词汇表的大小，即有多少个不同的词元。
        # embed_dim: 每个词元嵌入后的向量维度。
        # padding_idx (可选): 如果指定，该索引对应的嵌入向量会初始化为0，并且在训练中通常不更新。
        #                    这对于处理变长序列时的填充非常有用。
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.embed_dim = embed_dim # 保存嵌入维度，方便后续使用

    def forward(self, tokens: torch.Tensor):
        # tokens: 输入的词元ID序列，形状通常是 (batch_size, seq_len)
        # 输出: 词嵌入序列，形状是 (batch_size, seq_len, embed_dim)
        
        # 乘以 math.sqrt(self.embed_dim) 是一种常见的缩放技巧。
        # 在原始Transformer论文 "Attention Is All You Need" 中被提及，
        # 有助于在后续层（如点积注意力）中保持适当的方差，防止梯度过小或过大，
        # 使得模型训练更稳定。
        return self.embedding(tokens) * math.sqrt(self.embed_dim)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout) # Dropout层，用于正则化，防止过拟合

        # 创建一个足够长的位置编码矩阵 pe，形状为 (max_len, embed_dim)
        # max_len 是模型能处理的最大序列长度
        pe = torch.zeros(max_len, embed_dim)
        
        # 生成位置索引 (0, 1, ..., max_len-1)，并增加一个维度变为 (max_len, 1)
        # 例如，如果 max_len = 5, position = [[0], [1], [2], [3], [4]]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算除法项，用于缩放不同频率的正弦和余弦波
        # div_term 的思想是让不同维度上的波长呈几何级数变化
        # torch.arange(0, embed_dim, 2) 会生成 [0, 2, ..., embed_dim-2]
        # -math.log(10000.0) / embed_dim 是一个缩放因子
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        # 使用正弦函数填充偶数索引的维度 (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 使用余弦函数填充奇数索引的维度 (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 增加一个 batch 维度，变为 (1, max_len, embed_dim)，以方便后续与输入 (batch, seq_len, embed_dim) 相加时的广播
        pe = pe.unsqueeze(0)
        
        # 将 pe 注册为模型的 buffer。
        # buffer 是模型状态的一部分（会随模型一起保存和加载），但不是可训练参数（即梯度不会流过它们）。
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        # x: 词嵌入序列，形状 (batch_size, seq_len, embed_dim)
        
        # 从预计算的 pe 中取出与当前输入序列长度 (x.size(1)) 相匹配的部分，
        # 并将其加到输入 x 上。这里利用了PyTorch的广播机制。
        # self.pe[:, :x.size(1), :] 会截取 pe 的前 seq_len 个位置编码
        x = x + self.pe[:, :x.size(1), :]
        
        # 应用 dropout 后返回
        # 输出: 带有位置信息的嵌入序列，形状 (batch_size, seq_len, embed_dim)
        return self.dropout(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super(CausalSelfAttention, self).__init__()
        # embed_dim 必须能被 num_heads 整除
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # nn.MultiheadAttention: PyTorch实现的多头注意力机制。
        # embed_dim: 总的输入/输出特征维度。
        # num_heads: 注意力头的数量。多头允许模型在不同子空间中共同学习信息。
        # dropout: 应用于注意力权重图的dropout概率。
        # batch_first=True: 指定输入和输出张量的形状为 (batch_size, seq_len, embed_dim)，更符合常见习惯。
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        # x: 输入序列，形状 (batch_size, seq_len, embed_dim)
        # key_padding_mask (可选): 形状 (batch_size, seq_len)。
        #   用于指示输入序列中的哪些位置是填充（padding）的。
        #   如果一个位置为True，则该位置在注意力计算中会被忽略。
        
        device = x.device
        seq_len = x.size(1)
        causal_mask = generate_square_subsequent_mask(seq_len, device=device)

        # 对于自注意力 (Self-Attention)，query, key, value 都来自同一个输入 x。
        # is_causal=True (PyTorch >= 1.12): 这是关键！
        #   当设置为True时，`MultiheadAttention`会自动应用一个上三角掩码（causal mask），
        #   确保每个位置的query只能关注到key序列中当前及之前的位置。
        #   这样就实现了因果性，防止模型看到未来的信息。
        # attn_mask=None: 由于is_causal=True处理了因果掩码，这里通常设为None。
        #                 如果需要额外的自定义掩码（比如结合padding mask），可以更复杂地构造。
        # key_padding_mask: 传入我们之前生成的padding掩码，屏蔽padding token的影响。
        # need_weights=False: 如果我们不需要返回注意力权重图（通常在训练和推理中为了效率不返回），可以设为False。
        #                   如果需要分析注意力，可以设为True，此时会多一个attn_weights输出。
        attn_output, _ = self.mha(query=x, key=x, value=x,
                                  attn_mask=causal_mask,
                                  key_padding_mask=key_padding_mask,
                                  is_causal=True,
                                  need_weights=False)
        
        # attn_output 形状: (batch_size, seq_len, embed_dim)
        return attn_output

# 生成因果掩码（上三角掩码）的辅助函数
# 注意：当 nn.MultiheadAttention 的 is_causal=True 时，这个函数生成的掩码不是必需的。
def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    """为因果注意力生成一个方阵掩码。"""
    # torch.triu 创建一个上三角矩阵。diagonal=1 表示不包括主对角线。
    # 我们希望屏蔽未来的token，即上三角部分为 True。
    # 这会生成一个 sz x sz 的矩阵，其中上三角（不含对角线）为True，其余为False。
    # 例如 sz=3:
    # [[False,  True,  True],
    #  [False, False,  True],
    #  [False, False, False]]
    # 这正是 attn_mask 所需的格式，True 表示该位置在注意力计算中应被忽略。
    mask = torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)
    return mask


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        super(FeedForward, self).__init__()
        # 第一个线性层，将 embed_dim 扩展到 ff_dim (通常 ff_dim 是 embed_dim 的2到4倍)
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.dropout = nn.Dropout(dropout) # Dropout层
        # 第二个线性层，将 ff_dim 缩减回 embed_dim
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        # 激活函数，ReLU 是Transformer原始论文中使用的，GELU也是现代LLM中的常用选择
        self.activation = nn.ReLU() # 或者 nn.GELU()

    def forward(self, x: torch.Tensor):
        # x: 输入形状 (batch_size, seq_len, embed_dim)
        x = self.linear1(x)      # (batch_size, seq_len, ff_dim)
        x = self.activation(x)   # (batch_size, seq_len, ff_dim)
        x = self.dropout(x)      # (batch_size, seq_len, ff_dim)
        x = self.linear2(x)      # (batch_size, seq_len, embed_dim)
        # 输出形状: (batch_size, seq_len, embed_dim)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super(DecoderBlock, self).__init__()
        self.self_attention = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)
        
        # 层归一化 (Layer Normalization)
        # Transformer 原始论文中使用的是 Post-LN 结构 (LN 在残差连接之后)。
        # Pre-LN (LN 在自注意力/前馈网络之前) 也是一种常见的变体，有时能提供更稳定的训练。
        # 为了训练稳定，我们还是使用Pre-LN
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout 应用于子层的输出，在加入残差连接之前
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        # x: 输入形状 (batch_size, seq_len, embed_dim)
        # key_padding_mask (可选): 形状 (batch_size, seq_len), 用于CausalSelfAttention
        
        # 1. 因果自注意力子层 (Multi-Head Attention)
        residual = x # 保存输入 x 用于第一个残差连接
        # Pre-LN方案
        x = self.norm1(x)
        attn_output = self.self_attention(x, key_padding_mask=key_padding_mask)
        # 应用 dropout，然后进行残差连接 (Add) 和层归一化 (Norm)
        attn_output = self.dropout1(attn_output)
        x = residual + attn_output
        
        # 2. 前馈网络子层 (Feed Forward)
        residual = x # 更新残差连接的来源为上一层的输出
        # Pre-LN方案
        x = self.norm2(x)
        ff_output = self.feed_forward(x)
        # 再次应用 dropout，然后进行残差连接 (Add) 和层归一化 (Norm)
        ff_output = self.dropout2(ff_output)
        x = residual + ff_output
        
        # 输出形状: (batch_size, seq_len, embed_dim)
        return x

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, num_layers: int, vocab_size: int, embed_dim: int, 
                 num_heads: int, ff_dim: int, max_seq_len: int, 
                 dropout: float = 0.1, padding_idx: Optional[int] = 0,
                 tie_weights: bool = True): # 新增 tie_weights 参数
        super(DecoderOnlyTransformer, self).__init__()
        
        self.padding_idx = padding_idx # 保存padding_idx，用于生成padding_mask和传递给TokenEmbedding

        # 1. 词嵌入层
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim, padding_idx=self.padding_idx)
        # 2. 位置编码层
        self.positional_encoding = PositionalEncoding(embed_dim, dropout, max_seq_len)
        
        # 3. Decoder Block 堆栈
        # 使用 nn.ModuleList 来正确注册 DecoderBlock 列表中的模块，使其参数能被PyTorch自动管理。
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        
        # 4. 输出层 (也称为语言模型头, LM Head)
        # 将Decoder的最终输出（embed_dim维向量）映射回词汇表大小，得到每个词元的原始分数（logits）。
        # 后续可以通过Softmax将其转换为概率分布。
        self.output_layer = nn.Linear(embed_dim, vocab_size)
        
        # 5. (可选但推荐) 权重绑定 (Weight Tying)
        # 这是一个常见的技巧：共享词嵌入层 (self.token_embedding) 和输出层 (self.output_layer) 的权重矩阵。
        # 思想是：能够很好地表示一个词的向量，也应该能够很好地从隐藏状态预测出这个词。
        # 这可以显著减少模型参数数量，并有时能提高性能，特别是在词汇表较大时。
        if tie_weights:
            # 确保嵌入维度和输出层输入维度一致，这是权重绑定的前提
            if embed_dim != self.token_embedding.embedding.weight.size(1):
                raise ValueError("embed_dim must match embedding dim for weight tying")
            # 直接将输出层的权重指向嵌入层的权重
            self.output_layer.weight = self.token_embedding.embedding.weight

        # 初始化模型参数 (一个好的实践)
        self._init_weights()

    def _init_weights(self):
        # 对模型中的不同类型的层进行参数初始化是一种常见的实践，有助于模型训练。
        # 例如，Xavier/Glorot 初始化常用于线性层和嵌入层。
        for module in self.modules(): # self.modules() 会递归地返回模型中所有的模块
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight) # 使用 Xavier 均匀初始化权重
                if module.bias is not None:
                    nn.init.zeros_(module.bias) # 将偏置初始化为0
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight) # 使用 Xavier uniform 初始化嵌入权重
                if module.padding_idx is not None:
                    # 特别地，确保 padding_idx 对应的嵌入向量是零，并且在训练中（如果优化器不特殊处理）不会被更新。
                    module.weight.data[module.padding_idx].zero_() 
            elif isinstance(module, nn.LayerNorm):
                # LayerNorm 的 gamma (weight) 通常初始化为1，beta (bias) 初始化为0
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


    def _generate_padding_mask(self, src: torch.Tensor) -> Optional[torch.Tensor]:
        # src: 输入的词元ID序列，形状 (batch_size, seq_len)
        # 功能: 生成一个布尔掩码，标记出输入序列中的padding位置。
        #       True表示对应位置是padding，应该在注意力计算中被忽略。
        if self.padding_idx is None:
            return None # 如果没有定义padding_idx，则不生成掩码
        
        src_padding_mask = (src == self.padding_idx) # 形状 (batch_size, seq_len)
        # 例如: src = [[1, 2, 0], [3, 0, 0]], padding_idx = 0
        # mask =    [[F, F, T], [F, T, T]]
        return src_padding_mask

    def forward(self, src: torch.Tensor):
        # src: 输入的词元ID序列，形状 (batch_size, seq_len)
        #      例如: [[101, 1034, 203, 0, 0], [101, 405, 589, 382, 0]] (0是padding_idx)
        
        # 1. 生成 padding 掩码 (key_padding_mask)
        #    这个掩码会传递给CausalSelfAttention，用于在计算注意力分数时忽略padding token。
        src_key_padding_mask = self._generate_padding_mask(src) 
        # 形状: (batch_size, seq_len) 或者 None

        # 2. 词嵌入 和 位置编码
        # (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        x = self.token_embedding(src)      
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
        x = self.positional_encoding(x) 
        
        # 3. 通过 Decoder 块堆栈
        #    每一层 DecoderBlock 都会接收 src_key_padding_mask，
        #    其内部的 CausalSelfAttention 会使用 is_causal=True 自动处理因果掩码。
        for block in self.decoder_blocks:
            x = block(x, key_padding_mask=src_key_padding_mask) 
            # x 形状保持: (batch_size, seq_len, embed_dim)
            
        # 4. 输出层
        #    将Decoder最终的隐藏状态通过线性层映射到词汇表空间，得到logits。
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, vocab_size)
        logits = self.output_layer(x) 
        
        return logits # 返回每个位置上，对词汇表中每个词的预测分数


# 测试
if __name__ == '__main__':
    # 定义模型超参数
    num_layers = 12     # Decoder Block 的层数
    vocab_size = 25000  # 假设我们的词汇表大小为1000
    embed_dim = 768    # 嵌入维度
    num_heads = 12      # 注意力头的数量 (需要能整除embed_dim)
    ff_dim = embed_dim * 4 # 前馈网络内部维度 (通常是embed_dim的2到4倍)
    max_seq_len = 1024  # 模型能处理的最大序列长度 (用于位置编码)
    dropout_rate = 0.1 # Dropout比率
    padding_token_id = 0 # 假设 padding token 的 ID 是 0

    # 实例化模型
    model = DecoderOnlyTransformer(
        num_layers=num_layers,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        max_seq_len=max_seq_len,
        dropout=dropout_rate,
        padding_idx=padding_token_id,
        tie_weights=True # 启用权重绑定，共享输入嵌入和输出投影权重
    )
    
    # 打印模型结构
    print("模型结构:")
    print(model)
    
    # 统计模型可训练参数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"\n模型总可训练参数量: {num_params:,}M")

    # 创建一个模拟的输入批次
    batch_size = 4
    seq_len = 50  # 当前批次的序列长度，应 <= max_seq_len
    
    # 模拟输入 token IDs (随机整数，范围在 1 到 vocab_size-1 之间，0留给padding)
    input_tokens = torch.randint(1, vocab_size, (batch_size, seq_len)) # 先生成非 padding
    
    # 模拟padding: 让批次内序列长度不一致
    input_tokens[0, seq_len-5:] = padding_token_id  # 第一个序列末尾5个token是padding
    input_tokens[1, seq_len-10:] = padding_token_id # 第二个序列末尾10个token是padding
    # input_tokens 现在是 (batch_size, seq_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_tokens = input_tokens.to(device)

    # 模型前向传播
    # 如果是训练模式，应调用 model.train()。如果是评估/推理模式，调用 model.eval()
    # model.train() # 或 model.eval()
    
    print(f"\n输入Token序列形状: {input_tokens.shape}")
    logits = model(input_tokens) # 执行前向传播
    
    # 输出 logits 的形状应为 (batch_size, seq_len, vocab_size)
    print(f"输出Logits形状: {logits.shape}")

    # (可选) 检查logits中padding位置是否受影响
    # 理想情况下，padding token不应影响非padding token的表示，
    # 并且其自身的logit输出可能没有意义，或者在计算损失时会被忽略。
    # 例如，可以检查第一个样本的最后一个非padding token的logit与最后一个padding token的logit。
    if padding_token_id is not None and seq_len > 10:
        print("\n检查padding对logits的影响（简单示例）：")
        non_padding_logit_example = logits[0, seq_len-6, :5] # 第一个样本，倒数第6个token（非padding）的前5个logit
        padding_logit_example = logits[0, seq_len-1, :5]    # 第一个样本，最后一个token（padding）的前5个logit
        print(f"  某非Padding Token的Logits (前5): {non_padding_logit_example.detach().cpu().numpy()}")
        print(f"  某Padding Token的Logits (前5): {padding_logit_example.detach().cpu().numpy()}")
