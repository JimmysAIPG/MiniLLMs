import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
import math
import os
from functools import partial
from tqdm import tqdm
from torch.amp import GradScaler, autocast

from MiniLlmsModelV2 import DecoderOnlyTransformer
from sft_alpaca_dataset import SFTDataset, sft_collate_fn 


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 1. 超参数和配置 (SFT版本)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("xpu")
TOKENIZER_FILE = "bpe_tokenizer_from_arrow/tokenizer.json"
PRETRAINED_CHECKPOINT_PATH = "logs/checkpoints_bf16/best_model.pt" 

MAX_SEQ_LENGTH = 512
BATCH_SIZE = 64 # 根据自己的显卡显存选择合适的BATCH_SIZE，16GB建议设置为32
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
CLIP_GRAD_NORM = 1.0
GRADIENT_ACCUMULATION_STEPS = 4

EMBED_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 8
DROPOUT = 0.1
TIE_WEIGHTS = True
NUM_KV_HEADS = 4   # Key/Value heads (GQA)
FF_DIM = int(2/3 * 4 * EMBED_DIM) # SwiGLU 推荐维度

TENSORBOARD_LOG_DIR = "logs/sft_val/tensorboard"
CHECKPOINT_DIR="logs/sft_val/checkpoints"
LOG_INTERVAL = 10

USE_AMP = True
AMP_DTYPE = torch.bfloat16
USE_SCALER = True


def validate_sft_epoch(model, dataloader, criterion, device, epoch_num, writer):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_num+1} [SFT Validate]")

    with torch.no_grad():  # 不计算梯度
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            model_input = input_ids[:, :-1].contiguous()
            targets = labels[:, 1:].contiguous()

            with autocast(enabled=USE_AMP, dtype=AMP_DTYPE, device_type=device.type):
                logits = model(model_input)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss) if avg_loss > 0 else float('inf')
    
    writer.add_scalar('Loss/sft_validation_epoch', avg_loss, epoch_num)
    writer.add_scalar('Perplexity/sft_validation_epoch', perplexity, epoch_num)

    print(f"Epoch {epoch_num+1} Validation -> Avg Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
    return avg_loss


def train_sft_epoch(model, dataloader, optimizer, criterion, device, epoch_num, writer, global_step, scaler):
    model.train()
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch_num+1} [SFT Train]")
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in progress_bar:
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        
        # 模型的输入是到倒数第二个token的所有内容
        model_input = input_ids[:, :-1].contiguous()
        # 目标(labels)是从第二个token开始，与模型输出的logits对齐
        targets = labels[:, 1:].contiguous()
        
        with autocast(enabled=USE_AMP, dtype=AMP_DTYPE, device_type=device.type):
            logits = model(model_input)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = loss / GRADIENT_ACCUMULATION_STEPS

        if USE_SCALER:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        current_loss_item = loss.item() * GRADIENT_ACCUMULATION_STEPS

        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            if USE_SCALER:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % LOG_INTERVAL == 0:
                writer.add_scalar('Loss/sft_train_batch', current_loss_item, global_step)
            
            progress_bar.set_postfix(loss=f"{current_loss_item:.4f}", step=global_step)
    
    return global_step


def main():
    print(f"Using device: {DEVICE}")
    os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=TOKENIZER_FILE,
        pad_token='<pad>', bos_token='<s>', eos_token='</s>',
        unk_token='<unk>', mask_token='<mask>'
    )
    VOCAB_SIZE = tokenizer.vocab_size
    PADDING_IDX = tokenizer.pad_token_id

    # 1. 加载和准备 SFT 数据集
    print("Loading and splitting the dataset...")
    # dolly_dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    alpaca_json_path = "./openwebtext_arrow_dataset/sampled_dataset_3gb.jsonl"
    dolly_dataset = load_dataset('json', data_files=alpaca_json_path, split='train')
    
    # 关键改动：分割数据集
    train_val_split = dolly_dataset.train_test_split(test_size=0.1, seed=77)
    train_hf_dataset = train_val_split['train']
    val_hf_dataset = train_val_split['test']

    print(f"Train dataset size: {len(train_hf_dataset)}")
    print(f"Validation dataset size: {len(val_hf_dataset)}")
    
    train_dataset = SFTDataset(train_hf_dataset, tokenizer, MAX_SEQ_LENGTH)
    val_dataset = SFTDataset(val_hf_dataset, tokenizer, MAX_SEQ_LENGTH) # 创建验证集实例
    
    # 校验数据集
    print("\n--- Inspecting a sample from SFTDataset ---")
    sample = train_dataset[0] # 取第一个样本
    input_ids = sample['input_ids']
    labels = sample['labels']

    print("Full Input Text (decoded from input_ids):")
    print(f"'{tokenizer.decode(input_ids)}'")
    print("-" * 20)

    # 解码应该被计算Loss的部分
    response_labels = [token_id for token_id in labels if token_id != -100]
    print("Response Text (decoded from labels):")
    print(f"'{tokenizer.decode(response_labels)}'")
    print("-" * 20)

    # 检查 input_ids 和 labels 的长度是否一致
    print(f"Length of input_ids: {len(input_ids)}")
    print(f"Length of labels: {len(labels)}")
    print("-------------------------------------------\n")

    # 确认dolly数据集中的'response'字段没有问题，比如大量空字符串
    # 也可以加一个简单的检查
    # empty_responses = sum(1 for item in train_hf_dataset if not item['response'].strip())
    empty_responses = sum(1 for item in train_hf_dataset if not item['output'].strip())
    print(f"Found {empty_responses} empty responses in the training set.")

    # 2. 创建 DataLoader
    collate_fn = partial(sft_collate_fn, tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=4)
    
    # ... [模型初始化和加载预训练权重的代码和之前一样] ...
    model = DecoderOnlyTransformer(
        num_layers=NUM_LAYERS, vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
        ff_dim=FF_DIM, num_kv_heads=NUM_KV_HEADS,
        max_seq_len=MAX_SEQ_LENGTH, dropout=DROPOUT,
        padding_idx=PADDING_IDX, tie_weights=TIE_WEIGHTS
    ).to(DEVICE)
    
    print(f"Loading pretrained weights from: {PRETRAINED_CHECKPOINT_PATH}")
    checkpoint = torch.load(PRETRAINED_CHECKPOINT_PATH, map_location=DEVICE)
    model_state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(model_state_dict, strict=False) # 使用 strict=False 更稳健
    print("Pretrained weights loaded successfully.")
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100) 
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)
    scaler = GradScaler(enabled=USE_SCALER)
    writer = SummaryWriter(TENSORBOARD_LOG_DIR)
    
    # 3. 开始 SFT 训练，并加入验证逻辑
    print("Starting SFT with validation...")
    global_step = 0
    best_val_loss = float('inf') # 用于追踪最佳模型

    for epoch in range(NUM_EPOCHS):
        global_step = train_sft_epoch(model, train_dataloader, optimizer, criterion, DEVICE, epoch, writer, global_step, scaler)
        
        # 每个 epoch 结束后进行验证
        val_loss = validate_sft_epoch(model, val_dataloader, criterion, DEVICE, epoch, writer)
        
        # 检查是否是最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"✨ New best validation loss: {best_val_loss:.4f}. Saving best model...")
            best_model_path = os.path.join(CHECKPOINT_DIR, "sft_best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Best SFT model saved to {best_model_path}")

    writer.close()
    print(f"\nSFT complete. The best model is saved at {os.path.join(CHECKPOINT_DIR, 'sft_best_model.pt')}")

if __name__ == "__main__":
    main()