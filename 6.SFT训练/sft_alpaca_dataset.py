import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

class SFTDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer: PreTrainedTokenizerFast, max_seq_length: int):
        super().__init__()
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        # 定义我们的两种指令模板，这两种模板也完全适用于 Alpaca 数据集
        self.template_with_input = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
        )
        self.template_without_input = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n"
        )

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> dict:
        # 1. 从 Hugging Face dataset 中获取原始数据项
        item = self.hf_dataset[idx]
        instruction = item['instruction']
        response = item['output']

        # Alpaca 数据集中，上下文信息在 'input' 字段里
        context_input = item.get('input', None)

        # 根据 'input' 字段是否存在且不为空，选择不同的模板
        if context_input:
            prompt = self.template_with_input.format(instruction=instruction, input=context_input)
        else:
            prompt = self.template_without_input.format(instruction=instruction)
        
        # 2. 对 prompt 和 response 分别进行编码 (这部分逻辑保持不变)
        prompt = prompt.lower()
        response = response.lower()
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)

        # 3. 将 prompt 和 response 的 token ids 拼接，并添加 EOS token (这部分逻辑保持不变)
        input_ids = prompt_ids + response_ids + [self.tokenizer.eos_token_id]
        
        # 4. 创建 labels，这是 SFT 的核心 (这部分逻辑保持不变)
        # 将 prompt 部分的 label 设置为 -100，这样模型在计算 loss 时会忽略它们
        prompt_len = len(prompt_ids)
        labels = [-100] * prompt_len + response_ids + [self.tokenizer.eos_token_id]

        # 5. 截断到最大长度 (这部分逻辑保持不变)
        input_ids = input_ids[:self.max_seq_length]
        labels = labels[:self.max_seq_length]

        # 转换为 PyTorch 张量
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

# 它处理的是 __getitem__ 返回的字典，其结构并未改变
def sft_collate_fn(batch, tokenizer):
    input_ids_list = [item['input_ids'] for item in batch]
    labels_list = [item['labels'] for item in batch]

    padded_inputs = tokenizer.pad(
        {'input_ids': input_ids_list},
        padding=True,
        return_tensors='pt'
    )

    max_len = padded_inputs['input_ids'].size(1)
    # 手动 padding labels，因为 pad_token_id 默认是 0，而我们需要 -100 来忽略 loss
    padded_labels = torch.full((len(batch), max_len), fill_value=-100, dtype=torch.long)
    for i, labels in enumerate(labels_list):
        padded_labels[i, :len(labels)] = labels

    return {
        "input_ids": padded_inputs['input_ids'],
        "attention_mask": padded_inputs['attention_mask'],
        "labels": padded_labels
    }