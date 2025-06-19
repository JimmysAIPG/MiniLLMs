from datasets import load_from_disk, load_dataset
from tokenizers import ByteLevelBPETokenizer
import os

# 假设 ARROW_DATASET_DIR 是你保存 Arrow 数据集的路径
ARROW_DATASET_DIR = "./openwebtext_arrow_dataset/train"
BPE_TOKENIZER_SAVE_DIR = "./bpe_tokenizer_from_arrow"
VOCAB_SIZE = 15000  # BPE词表大小示例
MIN_FREQUENCY = 2   # 最小词频示例


def get_openwebtext_dataset(arrow_data_path):
    return load_from_disk(arrow_data_path)

def batch_iterator(dataset, text_column="text", batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size][text_column]

def train_bpe_from_arrow_dataset(arrow_dataset_path: str, save_dir: str):
    if not os.path.exists(arrow_dataset_path):
        print(f"错误: Arrow数据集路径不存在: {arrow_dataset_path}")
        return

    print(f"从Arrow数据集加载数据: {arrow_dataset_path}")
    dataset = get_openwebtext_dataset(arrow_dataset_path)
    print(f"数据集加载完成，样本数: {len(dataset)}")

    # 初始化并训练 BPE 分词器
    tokenizer = ByteLevelBPETokenizer(add_prefix_space=False)

    print("开始训练BPE分词器...")
    tokenizer.train_from_iterator(
        batch_iterator(dataset),
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ],
    )
    print("BPE分词器训练完成.")

    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
    print(f"BPE分词器模型已保存到: {save_dir}/tokenizer.json")

# 调用BPE训练
if __name__ == "__main__":
    train_bpe_from_arrow_dataset(ARROW_DATASET_DIR, BPE_TOKENIZER_SAVE_DIR)