import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast
import os
import argparse
from collections import OrderedDict

# 确保可以从同级目录导入你的模型定义
from MiniLlmsModel import DecoderOnlyTransformer 

# --- 1. 配置（必须与训练时完全一致！） ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("xpu")
TOKENIZER_FILE = "bpe_tokenizer_from_arrow/tokenizer.json"

# 模型参数 (必须与被加载的 checkpoint 的训练参数匹配)
EMBED_DIM = 512
NUM_HEADS = 8
FF_DIM = EMBED_DIM * 4
NUM_LAYERS = 8
MAX_SEQ_LENGTH = 512 # 模型知道的最大序列长度
DROPOUT = 0.1 # 虽然在 eval 模式下 dropout 不生效，但保持一致性是好习惯
TIE_WEIGHTS = True

def load_model_and_tokenizer(checkpoint_path: str):
    """
    加载 tokenizer 和从指定 checkpoint 恢复的模型。
    """
    print("--- 开始加载资源 ---")
    
    # a. 加载 Tokenizer
    print(f"加载 Tokenizer from: {TOKENIZER_FILE}")
    if not os.path.exists(TOKENIZER_FILE):
        raise FileNotFoundError(f"Tokenizer 文件未找到: {TOKENIZER_FILE}")
    
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=TOKENIZER_FILE,
        pad_token='<pad>', bos_token='<s>', eos_token='</s>',
        unk_token='<unk>', mask_token='<mask>'
    )
    vocab_size = tokenizer.vocab_size
    padding_idx = tokenizer.pad_token_id
    print(f"Tokenizer 加载完毕。词汇表大小: {vocab_size}")

    # b. 初始化模型结构 (使用与训练时相同的参数)
    print("初始化模型结构...")
    model = DecoderOnlyTransformer(
        num_layers=NUM_LAYERS,
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        max_seq_len=MAX_SEQ_LENGTH,
        dropout=DROPOUT,
        padding_idx=padding_idx,
        tie_weights=TIE_WEIGHTS
    )
    print("模型结构初始化完成。")

    # c. 加载 checkpoint 文件
    print(f"从 checkpoint 加载模型权重: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint 文件未找到: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # d. 提取并加载 model_state_dict
    # 你的训练脚本保存的是一个字典，我们需要其中的 'model_state_dict'
    model_state_dict = checkpoint['model_state_dict']

    # [重要] 处理可能的 'module.' 前缀
    # 如果你的模型在训练时使用了 DataParallel 或 DDP，state_dict 的键会以 'module.' 开头
    # 而我们这里加载的模型是没有被包裹的，所以需要去掉这个前缀。
    if all(key.startswith('module.') for key in model_state_dict.keys()):
        print("检测到 'module.' 前缀，正在移除...")
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            name = k[7:] # 移除 `module.`
            new_state_dict[name] = v
        model_state_dict = new_state_dict

    model.load_state_dict(model_state_dict)
    print("模型权重加载成功！")

    # e. 将模型移动到设备并设置为评估模式
    model.to(DEVICE)
    model.eval() # 非常重要！这会关闭 dropout 和其他训练特有的层
    
    print("--- 资源加载完毕 ---")
    return model, tokenizer

def generate_text(
    model: nn.Module, 
    tokenizer: PreTrainedTokenizerFast, 
    prompt: str, 
    max_new_tokens: int = 50,
    temperature: float = 0.8, # 用于控制生成文本的随机性
    top_k: int = 50 # 用于限制采样范围，增加文本质量
    ):
    """
    使用加载的模型生成文本。
    """
    print(f"\n--- 开始生成文本 ---")
    print(f"输入提示 (Prompt): '{prompt}'")
    
    # 1. 将输入文本编码为 token IDs
    # return_tensors='pt' 会返回 PyTorch 张量
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(DEVICE)
    
    # 2. 确保输入不会过长
    if input_ids.shape[1] >= MAX_SEQ_LENGTH:
        print(f"警告：输入长度 ({input_ids.shape[1]}) 已达到或超过模型最大长度 ({MAX_SEQ_LENGTH}). 可能会截断。")
        input_ids = input_ids[:, -MAX_SEQ_LENGTH+1:]


    # 3. 使用 torch.no_grad() 以提高性能并节省内存
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 获取模型输出的 logits
            # 只需传递当前的 token 序列
            outputs = model(input_ids)
            
            # logits 的形状是 (batch_size, sequence_length, vocab_size)
            # 我们只关心最后一个 token 的预测
            next_token_logits = outputs[:, -1, :]
            
            # 使用 temperature 缩放 logits
            next_token_logits = next_token_logits / temperature
            
            # Top-K 采样
            # 将不属于 top-k 的 token 的概率设置为 -inf
            top_k_values, _ = torch.topk(next_token_logits, top_k)
            kth_value = top_k_values[:, -1]
            indices_to_remove = next_token_logits < kth_value.unsqueeze(-1)
            next_token_logits[indices_to_remove] = float('-inf')

            # 从修改后的 logits 分布中采样
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            
            # 将新生成的 token ID 添加到输入序列中
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            
            # 如果生成了结束符 (EOS token)，则停止生成
            if next_token_id.item() == tokenizer.eos_token_id:
                print("\n[检测到结束符，停止生成]")
                break

    # 4. 将所有生成的 token IDs 解码回文本
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    print(f"--- 生成完成 ---")
    return generated_text

def main():
    # 使用 argparse 来方便地从命令行指定 checkpoint
    parser = argparse.ArgumentParser(description="使用训练好的模型进行文本生成测试")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True, 
        help="要加载的模型 checkpoint 文件路径, 例如: logs/checkpoints_bf16/checkpoint_step_10000.pt"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Java language is better ",
        help="用于开始文本生成的提示语"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="要生成的最大新 token 数量"
    )
    args = parser.parse_args()

    # 加载模型和 tokenizer
    model, tokenizer = load_model_and_tokenizer(args.checkpoint)
    
    # 生成文本
    full_text = generate_text(
        model, 
        tokenizer, 
        prompt=args.prompt, 
        max_new_tokens=args.max_new_tokens
    )
    
    print("\n\n===最终生成结果===")
    print(full_text)
    print("==================\n")

if __name__ == "__main__":
    main()