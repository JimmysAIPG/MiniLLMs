import os
from datasets import load_from_disk, Dataset, Features # Dataset 和 Features 用于处理空数据集情况

# --- 配置参数 ---
# 输入：原始完整数据集的路径
FULL_DATASET_PATH = "openwebtext_arrow_dataset/train"  # 修改为你的完整数据集路径

# 输出：分割后数据集的保存位置
OUTPUT_BASE_DIR = "split_openwebtext_custom_sample"  # 输出目录的名称
TRAIN_DATASET_OUTPUT_PATH = os.path.join(OUTPUT_BASE_DIR, "train")
VALIDATION_DATASET_OUTPUT_PATH = os.path.join(OUTPUT_BASE_DIR, "validation")


TRAIN_SAMPLE_PERCENTAGE = 0.03  # 例如：3% 的总数据用于训练
VALIDATION_SAMPLE_PERCENTAGE = 0.005 # 例如：0.5% 的总数据用于验证
# 剩余的 (1.0 - TRAIN_SAMPLE_PERCENTAGE - VALIDATION_SAMPLE_PERCENTAGE) 的数据将不会被使用

# 用于打乱的随机种子，确保结果可复现
RANDOM_SEED = 77

# 日志文件
LOG_FILE = 'logs/dataset_custom_splitting_log.txt'

# 设置 TOKENIZERS_PARALLELISM 以避免 huggingface tokenizers 的一些并行处理警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def log_message(message):
    """将消息记录到控制台和日志文件"""
    print(message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def create_empty_dataset_with_schema(schema: Features):
    """根据给定的 schema 创建一个空的 Hugging Face Dataset"""
    empty_data = {name: [] for name in schema.keys()}
    return Dataset.from_dict(empty_data, features=schema)

def main():
    # --- 1. 初始化设置 ---
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE) # 清除旧日志
        
    os.makedirs(TRAIN_DATASET_OUTPUT_PATH, exist_ok=True)
    log_message(f"训练数据集将保存到: {TRAIN_DATASET_OUTPUT_PATH}")
    os.makedirs(VALIDATION_DATASET_OUTPUT_PATH, exist_ok=True)
    log_message(f"验证数据集将保存到: {VALIDATION_DATASET_OUTPUT_PATH}")

    log_message(f"配置信息:")
    log_message(f"  完整数据集路径: {FULL_DATASET_PATH}")
    log_message(f"  输出主目录: {OUTPUT_BASE_DIR}")
    log_message(f"  训练集抽样百分比: {TRAIN_SAMPLE_PERCENTAGE*100:.2f}%")
    log_message(f"  验证集抽样百分比: {VALIDATION_SAMPLE_PERCENTAGE*100:.2f}%")
    log_message(f"  随机种子: {RANDOM_SEED}")

    # --- 2. 验证百分比设置 ---
    if not (0.0 <= TRAIN_SAMPLE_PERCENTAGE <= 1.0):
        log_message(f"错误: TRAIN_SAMPLE_PERCENTAGE ({TRAIN_SAMPLE_PERCENTAGE}) 必须在 0.0 和 1.0 之间。")
        return
    if not (0.0 <= VALIDATION_SAMPLE_PERCENTAGE <= 1.0):
        log_message(f"错误: VALIDATION_SAMPLE_PERCENTAGE ({VALIDATION_SAMPLE_PERCENTAGE}) 必须在 0.0 和 1.0 之间。")
        return
    
    total_requested_percentage = TRAIN_SAMPLE_PERCENTAGE + VALIDATION_SAMPLE_PERCENTAGE
    if total_requested_percentage > 1.0:
        log_message(f"错误: 训练集和验证集的总百分比 ({total_requested_percentage*100:.2f}%) 不能超过 100%。")
        return
    
    unused_percentage = 1.0 - total_requested_percentage
    # 使用一个小的 epsilon 来比较浮点数
    if unused_percentage > 1e-6: # 1e-6 是一个小的容差值
        log_message(f"提示: 数据集中将有 {unused_percentage*100:.2f}% 的数据未被用于训练或验证。")
    
    # --- 3. 加载完整数据集 ---
    log_message(f"开始从以下路径加载完整数据集: {FULL_DATASET_PATH}")
    try:
        full_hf_dataset = load_from_disk(FULL_DATASET_PATH)
    except Exception as e:
        log_message(f"错误: 从 '{FULL_DATASET_PATH}' 加载数据集失败。异常: {e}")
        log_message("请确保路径正确且包含有效的 Hugging Face Arrow 数据集。")
        return
    
    dataset_len = len(full_hf_dataset)
    log_message(f"完整数据集加载成功。文档总数: {dataset_len}")
    log_message(f"数据集特征: {full_hf_dataset.features}")

    if dataset_len == 0:
        log_message("加载的数据集为空。将创建空的训练集和验证集。")
        # 获取原始数据集的 schema 以创建空的、结构相同的 Dataset 对象
        schema = full_hf_dataset.features
        train_subset = create_empty_dataset_with_schema(schema)
        validation_subset = create_empty_dataset_with_schema(schema)
    else:
        # --- 4. 打乱并分割数据集 ---
        log_message(f"使用随机种子 {RANDOM_SEED} 打乱数据集...")
        # .shuffle() 返回一个新的 Dataset 对象，其中包含一个打乱的索引映射
        shuffled_dataset = full_hf_dataset.shuffle(seed=RANDOM_SEED)
        
        # 计算训练集和验证集的样本数量
        # 使用 int() 进行截断取整，这与 datasets.train_test_split 计算数量的方式类似
        num_train_samples = int(dataset_len * TRAIN_SAMPLE_PERCENTAGE)
        num_val_samples = int(dataset_len * VALIDATION_SAMPLE_PERCENTAGE)
        
        log_message(f"计算得到的训练样本数: {num_train_samples} "
                    f"(目标: {TRAIN_SAMPLE_PERCENTAGE*100:.2f}% / {dataset_len} 条)")
        log_message(f"计算得到的验证样本数: {num_val_samples} "
                    f"(目标: {VALIDATION_SAMPLE_PERCENTAGE*100:.2f}% / {dataset_len} 条)")

        # 确保选取的总样本数不超过数据集总长度
        # (这一步主要用于防止极小概率的浮点数问题，理论上之前的百分比检查已覆盖)
        if num_train_samples + num_val_samples > dataset_len:
            log_message(f"警告: 计算的训练样本数 ({num_train_samples}) + 验证样本数 ({num_val_samples}) "
                        f"总和为 {num_train_samples + num_val_samples}, 超过了数据集总长度 ({dataset_len})。")
            log_message(f"这可能指示百分比计算或极小数据集的问题。将尝试调整验证集数量。")
            # 优先保证训练集数量，然后调整验证集数量
            num_val_samples = dataset_len - num_train_samples
            if num_val_samples < 0: # 如果训练集本身就超了（理论上不可能发生）
                num_val_samples = 0
                num_train_samples = dataset_len 
            log_message(f"调整后的验证样本数: {num_val_samples}")


        log_message("选取训练子集...")
        train_indices = range(num_train_samples)
        train_subset = shuffled_dataset.select(train_indices)
        
        log_message("选取验证子集...")
        # 验证集的索引起始于训练集索引之后
        validation_start_index = num_train_samples
        validation_end_index = num_train_samples + num_val_samples
        validation_indices = range(validation_start_index, validation_end_index)
        validation_subset = shuffled_dataset.select(validation_indices)

    log_message(f"实际训练子集大小: {len(train_subset)} 条文档。")
    log_message(f"实际验证子集大小: {len(validation_subset)} 条文档。")

    # --- 5. 保存分割后的数据集 ---
    log_message(f"开始保存训练数据集到: {TRAIN_DATASET_OUTPUT_PATH}")
    try:
        train_subset.save_to_disk(TRAIN_DATASET_OUTPUT_PATH)
        log_message("训练数据集保存成功。")
    except Exception as e:
        log_message(f"错误: 保存训练数据集失败。异常: {e}")
        return

    log_message(f"开始保存验证数据集到: {VALIDATION_DATASET_OUTPUT_PATH}")
    try:
        validation_subset.save_to_disk(VALIDATION_DATASET_OUTPUT_PATH)
        log_message("验证数据集保存成功。")
    except Exception as e:
        log_message(f"错误: 保存验证数据集失败。异常: {e}")
        return

    log_message("\n--- 数据集自定义分割与保存流程完成 ---")
    log_message(f"训练数据已保存至: {os.path.abspath(TRAIN_DATASET_OUTPUT_PATH)}")
    log_message(f"验证数据已保存至: {os.path.abspath(VALIDATION_DATASET_OUTPUT_PATH)}")
    
    num_actually_used = len(train_subset) + len(validation_subset)
    num_unused = dataset_len - num_actually_used
    if num_unused > 0 :
        log_message(f"原始数据集中有 {num_unused} 条文档未被使用。")
    elif dataset_len > 0 and num_actually_used < dataset_len : # 处理因int()取整导致的小差异
        log_message(f"由于取整，原始数据集中有 {num_unused} 条文档未被使用。")


if __name__ == '__main__':
    main()