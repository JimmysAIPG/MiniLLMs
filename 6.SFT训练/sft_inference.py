import torch
from transformers import PreTrainedTokenizerFast

# 导入你的模型定义
from MiniLlmsModelV2 import DecoderOnlyTransformer

# --- 1. 配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device('xpu')
TOKENIZER_FILE = "bpe_tokenizer_from_arrow/tokenizer.json"
SFT_CHECKPOINT_PATH = "logs/sft_val/checkpoints/sft_best_model.pt"

# 模型参数 (必须与你训练时的一致)
EMBED_DIM = 512
NUM_HEADS = 8
NUM_KV_HEADS = 4   # Key/Value heads (GQA)
FF_DIM = int(2/3 * 4 * EMBED_DIM) # SwiGLU 推荐维度
NUM_LAYERS = 8
MAX_SEQ_LENGTH = 512
TIE_WEIGHTS = True
PADDING_IDX = 0 

# --- 2. 定义推理模板 (与 SFT 训练时保持一致) ---
template_with_input = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
template_without_input = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)

# --- 3. 加载模型和分词器 ---
def load_model_and_tokenizer():
    """加载模型和分词器"""
    print("Loading tokenizer...")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=TOKENIZER_FILE,
        pad_token='<pad>', bos_token='<s>', eos_token='</s>',
        unk_token='<unk>', mask_token='<mask>'
    )
    global PADDING_IDX
    PADDING_IDX = tokenizer.pad_token_id
    
    print("Loading model...")
    model = DecoderOnlyTransformer(
        num_layers=NUM_LAYERS,
        vocab_size=tokenizer.vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        ff_dim=FF_DIM,
        max_seq_len=MAX_SEQ_LENGTH,
        padding_idx=PADDING_IDX,
        tie_weights=TIE_WEIGHTS
    )
    
    if not SFT_CHECKPOINT_PATH or SFT_CHECKPOINT_PATH == "":
         raise ValueError("SFT_CHECKPOINT_PATH is not set.")

    print(f"Attempting to load state dict from: {SFT_CHECKPOINT_PATH}")
    state_dict = torch.load(SFT_CHECKPOINT_PATH, map_location=DEVICE)
    
    if all(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    print("Model and tokenizer loaded successfully.")
    
    return model, tokenizer

# --- 4. 生成函数 (已修改并增加诊断信息) ---
@torch.no_grad()
def generate_response(
    model, 
    tokenizer, 
    instruction: str, 
    input_text: str = None, 
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 50,
    repetition_penalty: float = 1.2
):
    """
    根据指令和可选的输入生成回答。
    """
    # 1. 根据有无 input 选择模板，并构建 prompt
    if input_text:
        prompt = template_with_input.format(instruction=instruction, input=input_text)
    else:
        prompt = template_without_input.format(instruction=instruction)

    # 2. 将 prompt 编码成 token ids (并转为小写，与训练保持一致)
    prompt = prompt.lower() # 保持与训练时一致
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    
    output_ids = input_ids.clone()
    
    for i in range(max_new_tokens):
        logits = model(output_ids)
        next_token_logits = logits[:, -1, :]

        if i > 0 and repetition_penalty != 1.0:
            # 获取已经生成的token ID (不包括prompt)
            generated_ids = output_ids[0, input_ids.shape[1]:]
            # 为这些ID创建惩罚
            # 对于已经生成的token，我们将它的logit除以惩罚系数（如果logit为正）
            # 或乘以惩罚系数（如果logit为负），效果都是降低其概率
            score = torch.gather(next_token_logits[0], 0, generated_ids)
            # 只惩罚正的 logits
            score = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            # 将修改后的分数放回原位
            next_token_logits[0].scatter_(0, generated_ids, score)

        next_token_logits = next_token_logits / temperature
        
        if top_k > 0:
            v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
            next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')

        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        if next_token.item() == tokenizer.eos_token_id:
            print(f"[DEBUG] EOS token ({tokenizer.eos_token_id}) generated at step {i+1}. Stopping generation.")
            break
            
        output_ids = torch.cat([output_ids, next_token], dim=1)
    
    # 在分割前，解码完整的文本并打印出来
    full_response_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\nFull decoded text (before splitting): \n---START---\n{full_response_text}\n---END---")

    # 使用小写的分割符，与 prompt.lower() 保持一致
    split_marker = "### response:\n"
    
    # 检查分割符是否存在
    if split_marker in full_response_text:
        response_part = full_response_text.split(split_marker)[-1].strip()
    else:
        print("[WARNING] Split marker not found in the generated text! The output might be incorrect.")
        # 如果找不到，尝试从prompt长度后开始解码，作为备用方案
        prompt_decoded_len = len(tokenizer.decode(input_ids[0], skip_special_tokens=True))
        response_part = full_response_text[prompt_decoded_len:].strip()

    print("="*54)
    
    # 打印最终输出
    print("\n" + "="*20 + " Model Output " + "="*20)
    print(response_part)
    print("="*54 + "\n")

    return response_part


# --- 5. 交互式主程序 (与之前相同) ---
def main():
    model, tokenizer = load_model_and_tokenizer()
    
    print("\n========================================")
    print("  SFT Model Interactive-Inference")
    print("========================================")
    print("Enter 'exit' or 'quit' to end.")
    print("For instructions with context, type your instruction, press Enter, then type the context.")
    print("For instructions without context, type your instruction and press Enter twice.")
    print("----------------------------------------\n")

    while True:
        try:
            instruction = input("### Instruction:\n> ")
            if instruction.lower() in ["exit", "quit"]:
                break
                
            input_text = input("### Input (optional, press Enter to skip):\n> ")
            if input_text.strip() == "":
                input_text = None

            generate_response(
                model=model,
                tokenizer=tokenizer,
                instruction=instruction,
                input_text=input_text,
                max_new_tokens=450,
                temperature=0.8,
                top_k=10,
                repetition_penalty=1.3
            )

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break

if __name__ == "__main__":
    main()
