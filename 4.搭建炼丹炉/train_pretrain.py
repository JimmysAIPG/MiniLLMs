import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset, Dataset, load_from_disk
import math
import os
from typing import Optional, Dict, List
from functools import partial
from collections import deque, OrderedDict
from tqdm import tqdm
from transformers import get_scheduler

# AMP (Automatic Mixed Precision)
from torch.amp import GradScaler, autocast

from MiniLlmsModel import DecoderOnlyTransformer
from StreamingOpenWebTextDataset import StreamingOpenWebTextDataset, custom_collate_fn

# 1. 超参数和配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("xpu")
TOKENIZER_FILE = "bpe_tokenizer_from_arrow/tokenizer.json"
TRAIN_DATASET_PATH = "split_openwebtext_custom_sample/train"
VAL_DATASET_PATH = "split_openwebtext_custom_sample/validation"

MAX_SEQ_LENGTH = 512
BATCH_SIZE = 64
NUM_EPOCHS = 6
CLIP_GRAD_NORM = 1.0
GRADIENT_ACCUMULATION_STEPS = 8
WARMUP_STEPS = 4000 // GRADIENT_ACCUMULATION_STEPS

# SCHEDULER_TYPE = "cosine"
SCHEDULER_TYPE = 'constant'
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

def validate_epoch(model, dataloader, criterion, device, current_step, writer,
                   max_batches: Optional[int] = None):
    model.eval()
    total_loss = 0
    num_batches = 0

    # `desc` 可以保持 epoch 信息，或者显示当前 step
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader) if hasattr(dataloader, '__len__') and len(dataloader) > 0 else None, desc=f"Validation at Step {current_step}")

    with torch.no_grad():
        for batch_idx, batch in progress_bar:
            if max_batches is not None and num_batches >= max_batches:
                break

            input_ids = batch['input_ids'].to(device, non_blocking=True)
            model_input = input_ids[:, :-1].contiguous()
            targets = input_ids[:, 1:].contiguous()

            if model_input.size(1) == 0:
                print(f"Skipping val batch {batch_idx} due to empty model_input (original seq_len: {input_ids.size(1)})")
                continue

            with autocast(enabled=USE_AMP, dtype=AMP_DTYPE, device_type=device.type):
                logits = model(model_input)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            total_loss += loss.item()
            num_batches += 1
            if isinstance(progress_bar, tqdm):
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
    if num_batches == 0:
        print("Warning: No batches were processed during validation.")
        return float('inf'), float('inf')

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    perplexity = math.exp(avg_loss) if avg_loss > 0 and not math.isinf(avg_loss) and not math.isnan(avg_loss) else float('inf')
    
    # 日志记录使用 current_step 而不是 epoch_num
    writer.add_scalar('Loss/validation_step', avg_loss, current_step)
    writer.add_scalar('Perplexity/validation_step', perplexity, current_step)
    return avg_loss, perplexity

def train_epoch(model, dataloader, optimizer, criterion, device, epoch_num, writer, global_step, scaler, 
                saved_checkpoints_queue, lr_scheduler, val_dataloader, best_val_loss,
                validation_subset_num_batches):
    model.train()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader) if hasattr(dataloader, '__len__') and len(dataloader) > 0 else None, desc=f"Epoch {epoch_num+1} [Train]")

    optimizer.zero_grad(set_to_none=True)
    for batch_idx, batch in progress_bar:
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        model_input = input_ids[:, :-1].contiguous()
        targets = input_ids[:, 1:].contiguous()

        if model_input.size(1) == 0:
            print(f"Skipping batch {batch_idx} due to empty model_input (original seq_len: {input_ids.size(1)})")
            continue

        with autocast(enabled=USE_AMP, dtype=AMP_DTYPE, device_type=device.type):
            logits = model(model_input)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = loss / GRADIENT_ACCUMULATION_STEPS
        
        if USE_SCALER:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        current_loss_item = loss.item() * GRADIENT_ACCUMULATION_STEPS 
        total_loss += current_loss_item
        num_batches += 1

        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            if USE_SCALER:
                # 在裁剪梯度前，需要先unscale
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
                optimizer.step()

            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if global_step > 0 and global_step % LOG_INTERVAL == 0:
                writer.add_scalar('Loss/train_batch', current_loss_item, global_step)
                if USE_SCALER:
                    writer.add_scalar('Misc/grad_scale', scaler.get_scale(), global_step)
                current_lr = lr_scheduler.get_last_lr()[0]
                writer.add_scalar('LearningRate/step', current_lr, global_step)

            if isinstance(progress_bar, tqdm):
                progress_bar.set_postfix(loss=f"{current_loss_item:.4f}", best_val_loss=f"{best_val_loss:.4f}")
            global_step += 1

            if global_step > 0 and global_step % SAVE_INTERVAL == 0:
                #  1. 保存常规的、带step的checkpoint (用于断点续训)
                os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                checkpoint_name = f"checkpoint_step_{global_step}.pt"
                checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
                
                model_state_to_save = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                
                save_dict = {
                    'epoch': epoch_num,
                    'global_step': global_step,
                    'model_state_dict': model_state_to_save,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': current_loss_item,
                }
                if USE_SCALER:
                    save_dict['scaler_state_dict'] = scaler.state_dict()

                torch.save(save_dict, checkpoint_path)
                print(f"\nSaved regular checkpoint to {checkpoint_path} at step {global_step}")
                
                # 管理checkpoint队列
                if saved_checkpoints_queue.maxlen is not None:
                    path_to_remove_on_disk = None
                    if len(saved_checkpoints_queue) == saved_checkpoints_queue.maxlen:
                        path_to_remove_on_disk = saved_checkpoints_queue.popleft()
                    saved_checkpoints_queue.append(checkpoint_path)
                    if path_to_remove_on_disk:
                        try:
                            if path_to_remove_on_disk != checkpoint_path:
                                os.remove(path_to_remove_on_disk)
                                print(f"Removed old checkpoint: {path_to_remove_on_disk}")
                        except OSError as e:
                            print(f"Error removing old checkpoint {path_to_remove_on_disk}: {e}")

                # 2. 在验证集上进行评估
                print(f"\n--- Running validation at global_step {global_step} ---")
                # 传入 global_step 用于 TensorBoard 日志记录
                val_loss, val_perplexity = validate_epoch(
                    model, val_dataloader, criterion, DEVICE, global_step, writer, validation_subset_num_batches
                )
                print(f"--- Validation at step {global_step}: Val Loss: {val_loss:.4f}, Perplexity: {val_perplexity:.2f} ---")
                
                # 切换回训练模式
                model.train()

                # 3. 检查是否为最佳模型并保存
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"✨ New best validation loss: {best_val_loss:.4f}. Saving best model...")
                    
                    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
                    # 使用和常规checkpoint相同的字典结构
                    best_model_save_dict = {
                        'epoch': epoch_num,
                        'global_step': global_step,
                        'model_state_dict': model_state_to_save, # 复用上面获取的 state_dict
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': best_val_loss, # 记录下当时的最佳loss
                    }
                    if USE_SCALER:
                        best_model_save_dict['scaler_state_dict'] = scaler.state_dict()

                    torch.save(best_model_save_dict, best_model_path)
                    print(f"✅ Best model saved to {best_model_path} at step {global_step}\n")

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss, global_step, best_val_loss


def main():
    print(f"Using device: {DEVICE}")
    os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 1. 加载Tokenizer
    print(f"Loading tokenizer from: {TOKENIZER_FILE}")
    if not os.path.exists(TOKENIZER_FILE):
        raise RuntimeError(f"Tokenizer file not found at {TOKENIZER_FILE}")
    
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=TOKENIZER_FILE,
        pad_token = '<pad>', bos_token = '<s>', eos_token = '</s>',
        unk_token = '<unk>', mask_token = '<mask>'
    )
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer does not have a pad_token_id. Ensure '<pad>' is a known token.")
    
    VOCAB_SIZE = tokenizer.vocab_size
    PADDING_IDX = tokenizer.pad_token_id
    print(f"Tokenizer loaded. Vocab size: {VOCAB_SIZE}, Padding ID: {PADDING_IDX}")

    # 2. 加载数据集
    print("Loading Hugging Face dataset...")
    raw_train_dataset = load_from_disk(TRAIN_DATASET_PATH)
    raw_val_dataset = load_from_disk(VAL_DATASET_PATH)
    print(f"Raw train dataset size: {len(raw_train_dataset)}")
    print(f"Raw val dataset size: {len(raw_val_dataset)}")

    # 3. 创建 Streaming Dataset 实例
    train_dataset = StreamingOpenWebTextDataset(
        hf_dataset=raw_train_dataset, tokenizer=tokenizer, max_seq_length=MAX_SEQ_LENGTH,
        text_column=TEXT_COLUMN, overlap_size=OVERLAP_SIZE, precompute_total_samples=PRECOMPUTE_SAMPLES
    )
    val_dataset = StreamingOpenWebTextDataset(
        hf_dataset=raw_val_dataset, tokenizer=tokenizer, max_seq_length=MAX_SEQ_LENGTH,
        text_column=TEXT_COLUMN, overlap_size=OVERLAP_SIZE, precompute_total_samples=PRECOMPUTE_SAMPLES
    )
    if PRECOMPUTE_SAMPLES:
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of validation samples: {len(val_dataset)}")

    collate_fn_with_tokenizer = partial(custom_collate_fn, tokenizer=tokenizer, max_seq_length=MAX_SEQ_LENGTH)

    # 4. 创建 DataLoader
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn_with_tokenizer,
        num_workers=NUM_DATALOADER_WORKERS, pin_memory=(DEVICE.type == 'cuda'),
        persistent_workers=(NUM_DATALOADER_WORKERS > 0)
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn_with_tokenizer,
        num_workers=NUM_DATALOADER_WORKERS, pin_memory=(DEVICE.type == 'cuda'),
        persistent_workers=(NUM_DATALOADER_WORKERS > 0)
    )

    # 5. 初始化模型、损失函数、优化器
    print("Initializing model...")
    model = DecoderOnlyTransformer(
        num_layers=NUM_LAYERS, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
        ff_dim=FF_DIM, max_seq_len=MAX_SEQ_LENGTH, dropout=DROPOUT, padding_idx=PADDING_IDX,
        tie_weights=TIE_WEIGHTS
    ).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized. Total trainable parameters: {total_params/1e6:.2f}M")

    criterion = nn.CrossEntropyLoss(ignore_index=PADDING_IDX)

    if SCHEDULER_TYPE == 'cosine':
        initial_lr = LEARNING_RATE
        print(f"Using 'cosine' scheduler with initial LR: {initial_lr}")
    elif SCHEDULER_TYPE == 'constant':
        initial_lr = CONSTANT_LEARNING_RATE
        print(f"Using 'constant' scheduler with fixed LR: {initial_lr}")
    else:
        raise ValueError(f"Unsupported SCHEDULER_TYPE: {SCHEDULER_TYPE}. Must be 'cosine' or 'constant'.")

    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0.1)
    scaler = GradScaler(enabled=USE_SCALER) 
    
    # 为了让调度器知道总的衰减周期，我们需要计算总的训练步数
    if PRECOMPUTE_SAMPLES:
        num_micro_batches_per_epoch = len(train_dataloader)
        # 计算有效更新步数
        num_update_steps_per_epoch = num_micro_batches_per_epoch // GRADIENT_ACCUMULATION_STEPS
        MAX_TRAIN_STEPS = NUM_EPOCHS * num_update_steps_per_epoch
        print(f"Total micro-batches per epoch: {num_micro_batches_per_epoch}")
        print(f"Total effective update steps calculated for scheduler: {MAX_TRAIN_STEPS}")
    else:
        # 如果是纯流式数据集，无法预知总步数。
        # 我们可以根据经验设定一个较大的总步数，例如你图上显示的 800k。
        MAX_TRAIN_STEPS = 800000 
        print(f"Total training steps estimated for scheduler: {MAX_TRAIN_STEPS}")

    # 创建调度器实例
    print(f"Initializing scheduler of type: {SCHEDULER_TYPE}")
    if SCHEDULER_TYPE == "cosine":
        lr_scheduler = get_scheduler(
            name="cosine", # 使用余弦退火策略
            optimizer=optimizer,
            num_warmup_steps=WARMUP_STEPS,
            num_training_steps=MAX_TRAIN_STEPS
        )
    elif SCHEDULER_TYPE == "constant":
        lr_scheduler = get_scheduler(
            name="constant", # 固定学习率
            optimizer=optimizer
        )

    # 6. 初始化 TensorBoard Writer
    writer = SummaryWriter(TENSORBOARD_LOG_DIR)

    # 断点续训逻辑
    global_step = 0
    start_epoch = 0 # 默认从第一个 epoch (索引0) 开始
    best_val_loss = float('inf')

    if RESUME_CHECKPOINT_PATH and os.path.exists(RESUME_CHECKPOINT_PATH):
        print(f"Resuming training from checkpoint: {RESUME_CHECKPOINT_PATH}")
        try:
            checkpoint = torch.load(RESUME_CHECKPOINT_PATH, map_location=DEVICE)
            
            # 恢复模型状态
            # 处理 DDP/DP 保存的模型 (键通常以 "module." 开头)
            model_state_dict = checkpoint['model_state_dict']
            current_model_is_wrapper = isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel))
            checkpoint_state_is_wrapper = all(key.startswith('module.') for key in model_state_dict.keys())

            if not current_model_is_wrapper and checkpoint_state_is_wrapper:
                print("Checkpoint from wrapped model, current model is not. Stripping 'module.' prefix.")
                new_state_dict = OrderedDict()
                for k, v in model_state_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                model_state_dict = new_state_dict
            elif current_model_is_wrapper and not checkpoint_state_is_wrapper:
                # 这通常不应该发生，因为我们会保存 model.module.state_dict()
                # 但为了鲁棒性，可以添加一个警告或尝试添加前缀
                print("Warning: Current model is wrapped, but checkpoint state is not. Attempting to load directly.")
            
            model.load_state_dict(model_state_dict)

            # 恢复优化器状态
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # 重要: 如果优化器参数（如学习率）在代码中被更改，
                # 而你想使用checkpoint中的学习率，需要确保优化器状态加载后，
                # 如果需要，再手动设置param_groups中的学习率。
                # AdamW等优化器的状态（如momentum buffers）会被正确恢复。
                if SCHEDULER_TYPE == 'constant':
                    print(f"Resumed from checkpoint, now forcing constant learning rate to {CONSTANT_LEARNING_RATE}")
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = CONSTANT_LEARNING_RATE
            else:
                print("Warning: No optimizer_state_dict found in checkpoint. Optimizer starts from scratch.")


            # 恢复 GradScaler 状态 (如果使用 AMP)
            if USE_SCALER:
                if 'scaler_state_dict' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                    print("GradScaler state loaded.")
                else:
                    print("Warning: AMP is enabled, but no scaler_state_dict found. Initializing new scaler.")
            
            # 恢复训练进度
            if 'global_step' in checkpoint:
                global_step = checkpoint['global_step']
            if 'epoch' in checkpoint:
                # checkpoint['epoch'] 是已完成的 epoch (0-indexed)
                # 下一个 epoch 从 checkpoint['epoch'] + 1 开始
                start_epoch = checkpoint['epoch'] + 1 
            
            print(f"Advancing learning rate scheduler to step {global_step}...")
            for _ in range(global_step):
                lr_scheduler.step()
            
            print(f"Successfully resumed. Next epoch: {start_epoch + 1}, Next global_step: {global_step + 1}")
            if 'loss' in checkpoint:
                 print(f"Checkpoint was saved at loss: {checkpoint['loss']:.4f}")
            
            best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
            if os.path.exists(best_model_path):
                print(f"Found existing best_model.pt. Loading its validation loss.")
                best_model_checkpoint = torch.load(best_model_path, map_location='cpu')
                if 'val_loss' in best_model_checkpoint:
                    best_val_loss = best_model_checkpoint['val_loss']
                    print(f"Resumed best validation loss to: {best_val_loss:.4f}")

        except Exception as e:
            print(f"Error loading checkpoint: {e}. Training will start from scratch.")
            global_step = 0
            start_epoch = 0
            if USE_SCALER: # 重置 scaler 以防状态损坏
                scaler = GradScaler(enabled=USE_SCALER)
    else:
        if RESUME_CHECKPOINT_PATH: # 路径指定了但文件不存在
            print(f"Checkpoint file not found at '{RESUME_CHECKPOINT_PATH}'. Training will start from scratch.")
        else: # 没有指定路径
            print("No resume checkpoint specified. Training will start from scratch.")
    
    # 初始化 checkpoint 管理队列
    # maxlen=0 或负数表示不限制，None 也表示不限制。
    # 如果 MAX_CHECKPOINT_TO_KEEP <= 0，则 maxlen 为 None (无界队列)
    # 否则 maxlen 为 MAX_CHECKPOINT_TO_KEEP
    queue_maxlen = MAX_CHECKPOINT_TO_KEEP if MAX_CHECKPOINT_TO_KEEP > 0 else None
    saved_checkpoints_queue = deque(maxlen=queue_maxlen)

    # 如果从checkpoint恢复，并且希望队列能感知到已存在的checkpoints:
    if RESUME_CHECKPOINT_PATH and queue_maxlen is not None:
        # 扫描CHECKPOINT_DIR，按step排序，填充队列到maxlen
        print(f"Scanning {CHECKPOINT_DIR} for existing checkpoints to manage queue...")
        existing_cps_files = [
            f for f in os.listdir(CHECKPOINT_DIR)
            if f.startswith("checkpoint_step_") and f.endswith(".pt")
        ]

        # 从文件名中解析 step 数并排序
        def get_step_from_filename(filename):
            try:
                return int(filename.split('_')[-1].split('.')[0])
            except:
                return -1 #无法解析的文件名排在前面或后面
        existing_cps_files.sort(key=get_step_from_filename)
        paths_to_add_to_queue = [os.path.join(CHECKPOINT_DIR, f) for f in existing_cps_files]

        # 如果已有的比maxlen多，只取最新的maxlen个
        if len(paths_to_add_to_queue) > queue_maxlen:
            paths_to_add_to_queue = paths_to_add_to_queue[-queue_maxlen:]
            # 此时，比这 queue_maxlen 个更旧的 checkpoint 理论上应该在上次运行时被删除了。
            # 但如果上次运行中断或逻辑错误，可能会有多余的。
            # 这里可以选择主动清理一下多余的旧文件，但这通常是上次运行的责任。
            # 为了简单，我们只初始化队列。
        for cp_path in paths_to_add_to_queue:
            saved_checkpoints_queue.append(cp_path) # deque 会自动处理，保持最新的

        print(f"Initialized checkpoint queue with {len(saved_checkpoints_queue)} most recent checkpoints from disk:")
        for cp in saved_checkpoints_queue:
            print(f"  - {cp}")

    
    # 7. 训练和验证循环
    print("Starting training...")
    if start_epoch >= NUM_EPOCHS:
        print(f"Start epoch ({start_epoch}) is already >= total num_epochs ({NUM_EPOCHS}). Training finished or adjust NUM_EPOCHS.")
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---") # epoch 是 0-indexed, 显示时 +1
        
        train_loss, global_step, best_val_loss = train_epoch(
            model, train_dataloader, optimizer, criterion, DEVICE, epoch, writer, global_step, 
            scaler, saved_checkpoints_queue, lr_scheduler,
            val_dataloader=val_dataloader,
            best_val_loss=best_val_loss,
            validation_subset_num_batches=VALIDATION_SUBSET_NUM_BATCHES
        )
        print(f"Epoch {epoch+1} Training Average Loss: {train_loss:.4f}")
        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        writer.add_scalar('LearningRate/epoch', optimizer.param_groups[0]['lr'], epoch)

        val_loss, val_perplexity = validate_epoch(
            model, val_dataloader, criterion, DEVICE, global_step, writer, None
        )
        print(f"Epoch {epoch+1} Validation Average Loss: {val_loss:.4f}, Perplexity: {val_perplexity:.2f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"\n✨ New best validation loss: {best_val_loss:.4f}. Saving best model...")
            
            best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
            
            # 准备保存的内容，与普通checkpoint类似
            model_state_to_save = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            save_dict = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model_state_to_save,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss, # 记录下当时的最佳loss
            }
            if USE_SCALER:
                save_dict['scaler_state_dict'] = scaler.state_dict()

            # 保存文件，会覆盖上一个 "best_model.pt"
            torch.save(save_dict, best_model_path)
            print(f"✅ Best model saved to {best_model_path}\n")
        
    writer.close()
    print("\nTraining complete.")
    print(f"TensorBoard logs saved to: {TENSORBOARD_LOG_DIR}")


if __name__ == "__main__":
    main()