
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

1）修改`ARROW_DATASET_DIR`指定为ARROW格式数据集的路径
2）执行`python bpe_tokenizer_train_with_arrow.py`即可开始训练

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

