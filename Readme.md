
对应的系列文章教程，请点[击这里](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=Mzk3NTA5NzEyNw==&action=getalbum&album_id=4000680701623418884&subscene=159&subscene=&scenenote=https%3A%2F%2Fmp.weixin.qq.com%2Fs%2FkgWCru0daBA9q5lLkmtMWw&nolastread=1#wechat_redirect)进行查看，如果对我的文章以及后续内容感兴趣，可以扫码关注哦。

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

4. 搭建炼丹炉

文件夹内我已经将之前编写的`dataset`类和`MiniLlmsModel`模型类一同复制过来，可以根据自己的需要修改`train_pretrain`脚本中的参数，然后直接通过如下指令运行，进行模型训练。

```sh
python train_pretrain.py
```

参数配置主要是这部分：

```python
# 1. 超参数和配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("xpu")
TOKENIZER_FILE = "bpe_tokenizer_from_arrow/tokenizer.json"
TRAIN_DATASET_PATH = "split_openwebtext_custom_sample/train"
VAL_DATASET_PATH = "split_openwebtext_custom_sample/validation"

MAX_SEQ_LENGTH = 512
BATCH_SIZE = 64 # 显存爆了的话，可以调整这个参数
NUM_EPOCHS = 6
CLIP_GRAD_NORM = 1.0
GRADIENT_ACCUMULATION_STEPS = 8
WARMUP_STEPS = 4000 // GRADIENT_ACCUMULATION_STEPS

# 学习率方案配置
SCHEDULER_TYPE = "cosine"
# SCHEDULER_TYPE = 'constant'
LEARNING_RATE = 5e-4
CONSTANT_LEARNING_RATE = 1.5e-4

# 模型参数
EMBED_DIM = 512
NUM_HEADS = 8
FF_DIM = EMBED_DIM * 4
NUM_LAYERS = 8
DROPOUT = 0.1
TIE_WEIGHTS = True

# 数据集参数
TEXT_COLUMN = 'text'
OVERLAP_SIZE = MAX_SEQ_LENGTH // 4
PRECOMPUTE_SAMPLES = True
NUM_DATALOADER_WORKERS = min(4, os.cpu_count())

# TensorBoard
TENSORBOARD_LOG_DIR = "logs/tensorboard_bf16"
LOG_INTERVAL = 50 // GRADIENT_ACCUMULATION_STEPS

# Checkpointing
CHECKPOINT_DIR="logs/checkpoints_bf16"
SAVE_INTERVAL = 10000 // GRADIENT_ACCUMULATION_STEPS # 每多少个 global_step 保存一次
MAX_CHECKPOINT_TO_KEEP = 99 # 保留最近的多少个checkpoint, 0 或 None 表示全部保留
VALIDATION_SUBSET_NUM_BATCHES = 5000

# 设置为你想恢复的checkpoint的路径，例如 "logs/checkpoints_bf16/checkpoint_step_1000.pt"
# 如果为 None，则从头开始训练
RESUME_CHECKPOINT_PATH: Optional[str] = None
# RESUME_CHECKPOINT_PATH = "logs/checkpoints_bf16/checkpoint_step_12500.pt"

# BF16/AMP Configuration
USE_AMP = True
AMP_DTYPE = torch.bfloat16
USE_SCALER = True
```

- bpe_tokenizer_from_arrow：BPE训练的词表，因为openwebtext数据集实在太大，不想从零开始的话，可以直接使用我的词表
- 训练数据集我已经上传到云盘，训练数据我抽取了openwebtext数据集中10%的数据进行训练，训练数据3GB左右。如果觉得太大，可以在训练的时候在这个基础上再进行缩减。数据集下载链接：链接：https://pan.quark.cn/s/b0996f64e429 提取码：exUb

