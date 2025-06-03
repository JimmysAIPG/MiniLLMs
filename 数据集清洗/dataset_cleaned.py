import os
import re
import unicodedata
from bs4 import BeautifulSoup
from datasets import load_dataset, Dataset, concatenate_datasets, Features, Value, DatasetDict
from tqdm.auto import tqdm
import trafilatura
from datasketch import MinHash, MinHashLSH
import pickle
import ftfy


# 配置参数
DATASET_NAME = "openwebtext"  # Hugging Face Hub上的数据集名称
# DATASET_NAME = "Elriggs/openwebtext-100k" # 小规模数据集测试
# DATASET_NAME = "stas/openwebtext-10k"  # 更小规模数据集测试
# DATASET_NAME = "/home/jimmy/SourceCode/dataset/openwebtext/plain_text" # 本地数据集
NUM_PROC = os.cpu_count()  # 使用的CPU核心数，加快处理速度
CLEANED_TEXT_DIR = "./openwebtext_cleaned"
ARROW_DATASET_DIR = "./openwebtext_arrow_dataset"
LOG_FILE = "logs/cleaning_log.txt"

# MinHash 去重参数
MINHASH_NUM_PERM = 128  # MinHash的排列数量，影响签名的精度和大小
MINHASH_JACCARD_THRESHOLD = 0.85  # Jaccard相似度阈值，高于此阈值的视为重复
MINHASH_SHINGLE_SIZE = 5 # 用于生成MinHash的shingle大小（字符n-gram）
MAP_BATCH_SIZE = 1000 # 适当的批处理大小可以提高 .map() 效率


# 辅助函数
def log_message(message):
    print(message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")


# 1. 数据集下载与初步加载
def download_and_load_data():
    log_message(f"开始下载并加载数据集: {DATASET_NAME}")
    try:
        # OpenWebText 比较大，可能需要较长时间
        # 如果网络不好，可以尝试 "Elriggs/openwebtext-100k" 或者 "stas/openwebtext-10k" 作为小规模测试
        dataset = load_dataset(DATASET_NAME, trust_remote_code=True)
        return dataset
    except Exception as e:
        log_message(f"错误：下载或加载数据集失败: {e}")
        log_message("请检查网络连接或数据集名称是否正确。")
        log_message("你可以尝试使用一个小的数据集进行测试，例如 'wikitext', 'wikitext-2-raw-v1'。")
        log_message("或者OpenWebText的子集 'Elriggs/openwebtext-100k' (如果可用)。")
        return None


def remove_html(text):
    # 尝试使用 trafilatura，如果失败或结果为空，则用BeautifulSoup
    extracted_text = trafilatura.extract(text, include_comments=False, include_tables=False, output_format='txt')
    if extracted_text and extracted_text.strip():
        return extracted_text
    # 回退到BeautifulSoup
    soup = BeautifulSoup(text, "lxml")
    return soup.get_text()


def normalize_char(text):
    #    a. 字符白名单：移除不在预定义集合中的字符。
    #       将不希望的字符替换为空格
    text = re.sub(r'[^a-z0-9\s\.,!?;:\'\"\(\)&\+\#\@\_\%\/-]', ' ', text)

    #    b. 处理重复标点：将连续的多个相同标点符号规范化。
    #       例如："!!!!" -> "!", "..." -> " ... "
    #       移除开头连续出现的点号和空白字符
    text = re.sub(r'^[\s.]+', '', text)
    #       例如： hello,,,world -> hello, world
    text = re.sub(r'\s*([,.!? জানে;:"])\1{2,}\s*', r'\1 ', text)
    #       标准化省略号，例如 "..." 或 ". . ." 替换为 "..."
    text = re.sub(r'\s*(\.\s*){3,}\s*', '...', text)  

    #    c. 处理特定重复的特殊字符序列，例如过长的破折号或下划线
    text = re.sub(r'(-)\1{2,}', r'\1\1', text)
    text = re.sub(r'(_)\1{2,}', r'\1\1', text)

    return text

# 2. 数据清洗
def clean_text_sample(text):
    if not isinstance(text, str):
        return ""  # 处理非字符串输入

    # 2.1 移除开头连续出现的点号和空白字符
    text = re.sub(r'^[\s.]+', '', text)

    # 2.2 Unicode标准化
    text = ftfy.fix_text(text=text)
    text = unicodedata.normalize('NFKC', text)

    # 2.3 移除HTML标签
    text = remove_html(text=text)

    # 2.4 移除URLs和Email地址
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # 移除URL
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b', '', text)  # 移除Email

    # 2.5 大小写转换 (转为小写)
    text = text.lower()

    # 2.6 替换字符
    text = normalize_char(text=text)

    # 2.7 空白符处理
    text = re.sub(r'\s+', ' ', text).strip()  # 将多个空白符合并为一个空格，并去除首尾空格

    return text


def filter_text_sample(text_sample):
    # 过滤掉清洗后为空或过短的文本
    text = text_sample["text"]
    return len(text) > 50 and len(text) < 30000  # 长度过滤阈值，可调整


def save_cleaned_text_to_files(dataset, directory):
    log_message(f"开始保存清洗后的文本到目录: {directory}")
    os.makedirs(directory, exist_ok=True)

    # 为了方便BPE训练器读取，可以保存为多个txt文件或一个大txt文件
    # 这里选择保存为一个大文件，每行一个文档
    output_file_path = os.path.join(directory, "all_cleaned_text.txt")

    batch_size = 1000  # 每次写入的文档数
    with open(output_file_path, "w", encoding="utf-8") as f:
        for i in tqdm(range(0, len(dataset), batch_size), desc="Saving cleaned text"):
            batch = dataset[i:i + batch_size]
            for text_sample in batch['text']:
                f.write(text_sample + "\n")
    log_message(f"所有清洗后的文本已保存到: {output_file_path}")
    return [output_file_path]  # 返回文件路径列表，BPE训练器需要


def save_dataset_to_arrow(dataset: Dataset, save_path: str):
    """将Hugging Face Dataset对象保存为Arrow格式。"""
    log_message(f"开始将数据集以Arrow格式保存到: {save_path}")
    try:
        dataset.save_to_disk(save_path)
        log_message(f"数据集已成功保存到: {save_path}")
    except Exception as e:
        log_message(f"错误：保存数据集到Arrow格式失败: {e}")


# MinHash 相关函数
def get_shingles(text, k=MINHASH_SHINGLE_SIZE):
    """从文本生成 k-shingles (字符 n-grams)"""
    if not text or len(text) < k: # 确保文本长度足以生成shingles
        return set()
    return set(text[i:i+k] for i in range(len(text) - k + 1))

# 这个函数将在 .map() 中被多进程调用
def _compute_minhash_for_sample(text_sample_content):
    """为单个文本内容计算MinHash对象。"""
    if not text_sample_content:
        return None
    m = MinHash(num_perm=MINHASH_NUM_PERM)
    shingles = get_shingles(text_sample_content, k=MINHASH_SHINGLE_SIZE)
    if not shingles: # 如果文本太短或无法生成shingles
        return None
    for s in shingles:
        m.update(s.encode('utf-8'))
    return pickle.dumps(m)

def _compute_minhash_batch(batch_of_texts):
    """
    为一批文本计算MinHash Pickle对象。
    'batch_of_texts' 是一个字典，例如 {'text': ['text1', 'text2', ...]}
    返回一个字典，例如 {'minhash_pickled': [minhash_pickled_obj1, minhash_pickled_obj2, ...]}
    """
    minhashes_pickled = []
    for text_content in batch_of_texts['text']:
        minhashes_pickled.append(_compute_minhash_for_sample(text_content))
    return {'minhash_pickled': minhashes_pickled}


def apply_cleaning_and_deduplication_minhash(dataset):
    log_message("开始数据清洗流程...")

    # 1. 初步清洗文本内容
    cleaned_dataset = dataset.map(
        lambda example: {'text': clean_text_sample(str(example.get('text', '')))},
        num_proc=NUM_PROC, # 使用多进程
        # load_from_cache_file=False,
        desc="Cleaning text content"
    )
    log_message(f"文本内容初步清洗完成. 当前样本数: {len(cleaned_dataset)}")

    # 2. 过滤文本长度
    original_count_before_len_filter = len(cleaned_dataset)
    cleaned_dataset = cleaned_dataset.filter(
        filter_text_sample,
        num_proc=NUM_PROC, # 使用多进程
        desc="Filtering text by length"
    )
    log_message(f"文本长度过滤完成. 移除了 {original_count_before_len_filter - len(cleaned_dataset)} 个样本. 剩余样本数: {len(cleaned_dataset)}")

    if len(cleaned_dataset) == 0:
        log_message("警告: 长度过滤后数据集为空，跳过MinHash去重。")
        return cleaned_dataset

    log_message(f"开始使用MinHash进行去重 (num_perm={MINHASH_NUM_PERM}, threshold={MINHASH_JACCARD_THRESHOLD})...")
    log_message(f"去重前样本数量 (长度过滤后): {len(cleaned_dataset)}")

    # 阶段1: 并行计算所有文档的MinHash签名
    # 使用 .map() 并行处理，这会利用 NUM_PROC 个进程
    log_message(f"阶段1: 开始并行计算MinHash签名 (使用 {NUM_PROC} 个进程)...")
    
    # 定义输出特性，确保只包含序列化后的MinHash
    # 这是关键的优化步骤：创建一个只包含MinHash签名的新数据集
    minhash_features = Features({
        'minhash_pickled': Value('binary') # 'binary' 用于存储pickle序列化后的字节串
    })

    # dataset.map() 在使用 num_proc > 1 时，传递给函数的对象需要能被pickle序列化。
    # MinHash 对象本身通常是可以pickle的。
    dataset_with_pickled_minhashes = cleaned_dataset.map(
        _compute_minhash_batch, # 应用于批处理的函数
        batched=True,
        batch_size=MAP_BATCH_SIZE, # 每个进程一次处理的批大小
        num_proc=NUM_PROC,
        remove_columns=cleaned_dataset.column_names, # 移除所有来自 cleaned_dataset 的列
        features=minhash_features,                   # 指定新数据集的特性（列和类型）
        desc="Generating and Pickling MinHashes"
    )
    log_message("MinHash签名计算和序列化完成.")

    lsh = MinHashLSH(threshold=MINHASH_JACCARD_THRESHOLD, num_perm=MINHASH_NUM_PERM)
    unique_docs_original_indices = []  # 存储LSH判断为唯一的文档在cleaned_dataset中的索引
    indices_with_none_or_error_minhash = [] # 存储MinHash为None或处理出错的文档索引

    log_message("阶段2: 迭代处理MinHash并构建LSH索引...")

    # 直接迭代 dataset_with_pickled_minhashes 对象
    # original_idx 对应 cleaned_dataset 和 dataset_with_pickled_minhashes 中的原始顺序
    for original_idx, item in enumerate(tqdm(dataset_with_pickled_minhashes, desc="Processing MinHashes and LSH")):
        pickled_mh = item['minhash_pickled'] # 获取当前项的序列化MinHash

        if pickled_mh is not None:
            try:
                mh = pickle.loads(pickled_mh)
                # 查询LSH中与当前文档相似的文档
                # LSH query 返回的是之前插入的 key (这里是文档的原始索引的字符串形式)
                similar_docs_keys = lsh.query(mh)

                if not similar_docs_keys:
                    # 没有找到相似的已索引文档，这个文档是唯一的（到目前为止）
                    lsh.insert(str(original_idx), mh) # 将当前文档的MinHash加入LSH索引
                    unique_docs_original_indices.append(original_idx) # 保留这个文档的原始索引
                # else: 是重复文档，不加入 unique_docs_original_indices
            except pickle.UnpicklingError as e:
                log_message(f"警告: 无法反序列化索引 {original_idx} 处的MinHash: {e}. 将保留此文档。")
                indices_with_none_or_error_minhash.append(original_idx)
            except Exception as e: # 捕获其他可能的MinHash相关错误
                log_message(f"警告: 处理索引 {original_idx} 处的MinHash时发生错误: {e}. 将保留此文档。")
                indices_with_none_or_error_minhash.append(original_idx)
        else:
            # MinHash 为 None (例如文本过短无法生成shingles)
            indices_with_none_or_error_minhash.append(original_idx)
    
    # 合并LSH判定的唯一项和那些无法生成MinHash的项（如果策略是保留它们）
    final_indices_to_keep = sorted(list(set(unique_docs_original_indices + indices_with_none_or_error_minhash)))
    # 如果策略是仅保留LSH判定的唯一项，则:
    # final_indices_to_keep = sorted(unique_docs_original_indices)

    num_lsh_unique = len(unique_docs_original_indices)
    num_none_minhash = len(indices_with_none_or_error_minhash)

    log_message(f"MinHash去重完成. 保留了 {len(final_indices_to_keep)} 个样本。")
    log_message(f"（其中 {num_lsh_unique} 个通过LSH判定为唯一，{num_none_minhash} 个因无法生成MinHash而保留）。")
    log_message(f"共移除了 {len(cleaned_dataset) - len(final_indices_to_keep)} 个样本。")

    if not final_indices_to_keep:
        log_message("警告: MinHash去重后没有剩余样本。请检查参数或数据。")
        return cleaned_dataset.select([]) # Schema会保留

    deduplicated_dataset = cleaned_dataset.select(final_indices_to_keep)

    log_message(f"清洗和MinHash去重后样本数量: {len(deduplicated_dataset)}")
    if len(deduplicated_dataset) > 0:
        log_message(f"前5条样本示例 (清洗和去重后):")
        for i in range(min(5, len(deduplicated_dataset))):
            log_message(f"样本 {i}: {deduplicated_dataset[i]['text'][:200]}...")
    
    return deduplicated_dataset


# 主流程
if __name__ == "__main__":
    # 清空日志文件
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    log_message("--- LLM 数据处理与训练流程开始 ---")

    # 1. 下载和加载
    raw_dataset = download_and_load_data()

    if (isinstance(raw_dataset, DatasetDict)):
        log_message(f"数据集 {DATASET_NAME} 包含以下拆分: {list(raw_dataset.keys())}")
        for split_name, raw_dataset_split in raw_dataset.items():
            log_message(f"\n拆分处理: {split_name}")
            log_message(f"数据集加载完成. 样本数量: {len(raw_dataset_split)}")
            log_message(f"前5条样本示例 (原始):")
            for i in range(min(5, len(raw_dataset_split))):
                log_message(f"样本 {i}: {raw_dataset_split[i]['text'][:200]}...")  # 只显示前200字符

            if raw_dataset_split:
                # 2. 数据清洗
                cleaned_dataset = apply_cleaning_and_deduplication_minhash(raw_dataset_split)
                # 3. 数据集存储
                #数据集存储为Arrow格式
                save_path = f"{ARROW_DATASET_DIR}/{split_name}"
                os.makedirs(save_path, exist_ok=True) # 确保父目录存在
                save_dataset_to_arrow(cleaned_dataset, save_path)

                # 存成一个大的txt文件
                # cleaned_text_files = save_cleaned_text_to_files(cleaned_dataset, CLEANED_TEXT_DIR)
            else:
                log_message("未能加载数据集，流程终止。")

    log_message("--- LLM 数据处理流程结束 ---")
