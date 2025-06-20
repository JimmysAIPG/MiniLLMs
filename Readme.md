
1. 数据集清洗

分成两个代码脚本文件，一个是数据集清洗，一个是数据集采样切割，默认使用数据集为`OpenWebText`，脚本可直接运行

1.1 数据集清洗

```bash
python dataset_cleaned.py
```

1.2 数据集采样

```bash
python dataset_split.py
```

2. BPE训练与数据集加载管道搭建

BPE训练使用方法：

- 修改`ARROW_DATASET_DIR`指定为ARROW格式数据集的路径
- 执行`python bpe_tokenizer_train_with_arrow.py`即可开始训练

数据集加载的典型使用方法

```py
train_streaming_dataset = StreamingOpenWebTextDataset(
    hf_dataset=train_hf_subset,
    tokenizer=tokenizer,
    max_seq_length=MAX_SEQ_LENGTH,
    overlap_size=MAX_SEQ_LENGTH // 4,
    text_column='text', # 确保这里的列名与你的Arrow数据集中的文本列名一致
    precompute_total_samples=True
)

# 使用 functools.partial 将 tokenizer 和 max_seq_length 绑定到 collate_fn
from functools import partial
collate_fn_with_tokenizer = partial(custom_collate_fn, tokenizer=tokenizer, max_seq_length=MAX_SEQ_LENGTH)

train_data_loader = DataLoader(
    train_streaming_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn_with_tokenizer,
    pin_memory=True # 如果使用 GPU，有助于加速数据传输 [13]
    )
```

3. 模型搭建

模型典型的搭建方式如下，可以参考`MiniLlmsModel.py`内的`main`方法

```
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
```