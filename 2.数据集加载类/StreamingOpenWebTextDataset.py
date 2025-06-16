from torch.utils.data import IterableDataset
from datasets import Dataset
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
import torch
import os

class StreamingOpenWebTextDataset(IterableDataset):
    def __init__(self, hf_dataset: Dataset, tokenizer: PreTrainedTokenizerFast, max_seq_length: int, 
                 text_column: str = 'text', overlap_size: int = 128,
                 precompute_total_samples: bool = False,
                 num_proc_for_counting: int = None,
                 total_sample: int = None):
        super().__init__()
        self.hf_dataset = hf_dataset

        # 使用PreTrainedTokenizerFast加载tokenizer
        self.tokenizer = tokenizer

        # 确保tokenizer有pad_token
        if self.tokenizer.pad_token is None:
            raise ValueError("tokenizer pad token is none")

        self.max_seq_length = max_seq_length
        self.text_column = text_column

        if overlap_size >= max_seq_length:
            raise ValueError("overlap_size 必须小于 max_seq_length。")

        self.overlap_size = overlap_size
        self.stride = self.max_seq_length - self.overlap_size
        if self.stride <= 0:
            # stride必须为正，否则滑动窗口无法前进
            raise ValueError(f"Stride ({self.stride}) 必须为正。请检查 max_seq_length ({self.max_seq_length}) 和 overlap_size ({self.overlap_size})。")
        
        if total_sample is None:
            self._total_samples = None # 用于存储计算出的总样本数
        else:
            self._total_samples = total_sample
        if precompute_total_samples:
            print("Precomputing total number of samples. This may take a while for large datasets...")
            if num_proc_for_counting is None:
                num_proc_for_counting = os.cpu_count()
            self._compute_total_samples(num_proc_for_counting=num_proc_for_counting)
            print(f"Total number of samples precomputed: {self._total_samples}")
    
    def _tokenize_and_chunk_document_stream(self, document_text: str):
        """
        辅助函数：对单个文档进行分词和分块，并生成块。
        """
        if not document_text or not isinstance(document_text, str):
            return

        encoded_document = self.tokenizer(
            document_text,
            add_special_tokens=False,
            return_attention_mask=False,
            truncation=False,
            padding=False
        )
        doc_token_ids = encoded_document['input_ids']

        if not doc_token_ids:
            return

        max_covered_exclusive_end_idx_in_doc = 0
        for i in range(0, len(doc_token_ids), self.stride):
            chunk_token_ids = doc_token_ids[i : i + self.max_seq_length]

            if not chunk_token_ids:
                continue

            current_chunk_actual_exclusive_end_idx = i + len(chunk_token_ids)
            if current_chunk_actual_exclusive_end_idx <= max_covered_exclusive_end_idx_in_doc:
                continue

            yield {"input_ids": chunk_token_ids}
            max_covered_exclusive_end_idx_in_doc = current_chunk_actual_exclusive_end_idx

    def _count_chunks_in_document_text(self, document_text: str) -> int:
        """
        计算单个文档文本可以产生的 chunk 数量。
        这个方法不 yield，而是直接返回计数。
        """
        count = 0
        if not document_text or not isinstance(document_text, str):
            return 0

        # 注意：这里的 tokenizer 调用与 _tokenize_and_chunk_document_stream 中的一致
        encoded_document = self.tokenizer(
            document_text,
            add_special_tokens=False,
            return_attention_mask=False,
            truncation=False,
            padding=False
        )
        doc_token_ids = encoded_document['input_ids']

        if not doc_token_ids:
            return 0

        max_covered_exclusive_end_idx_in_doc = 0
        for i in range(0, len(doc_token_ids), self.stride):
            chunk_token_ids = doc_token_ids[i : i + self.max_seq_length]
            if not chunk_token_ids:
                continue
            current_chunk_actual_exclusive_end_idx = i + len(chunk_token_ids)
            if current_chunk_actual_exclusive_end_idx <= max_covered_exclusive_end_idx_in_doc:
                continue
            count += 1
            max_covered_exclusive_end_idx_in_doc = current_chunk_actual_exclusive_end_idx
        return count

    def _map_function_to_count_chunks(self, example: dict) -> dict:
        """
        这是传递给 hf_dataset.map() 的函数。
        它从 example 中提取文本，并使用 _count_chunks_in_document_text 计算 chunk 数。
        """
        document_text = example[self.text_column]
        num_chunks = self._count_chunks_in_document_text(document_text)
        return {"num_chunks_for_doc": num_chunks} # 返回包含新列的字典

    def _map_function_to_count_chunks_batched(self, examples: dict) -> dict:
        """
        这是传递给 hf_dataset.map(batched=True) 的函数。
        它从 examples 中提取一批文本，并计算每个文本的 chunk 数。
        """
        document_texts = examples[self.text_column]
        num_chunks_list = []
        for text in document_texts:
            num_chunks_list.append(self._count_chunks_in_document_text(text))
        return {"num_chunks_for_doc": num_chunks_list}

    def _compute_total_samples(self, num_proc_for_counting: int = None, use_batched_map: bool = True, map_batch_size: int = 1000):
        """
        使用 dataset.map() 并行计算总样本数。
        """
        if self._total_samples is not None:
            return self._total_samples

        if num_proc_for_counting is None:
            num_proc_for_counting = os.cpu_count()
            if num_proc_for_counting is None: # os.cpu_count() might return None
                num_proc_for_counting = 1
            print(f"Number of processors for counting not specified, defaulting to {num_proc_for_counting}.")
        
        if num_proc_for_counting > os.cpu_count():
             print(f"Warning: num_proc_for_counting ({num_proc_for_counting}) is greater than available CPUs ({os.cpu_count()}). Setting to {os.cpu_count()}.")
             num_proc_for_counting = os.cpu_count()


        print(f"Calculating total samples using dataset.map() with num_proc={num_proc_for_counting}...")

        # `datasets.map()` 需要一个函数，该函数接收一个 example (dict) 并返回一个 dict。
        # `self` (包含 tokenizer, stride 等) 会被 pickle 并传递给子进程。
        # PreTrainedTokenizerFast 通常是可 pickle 的。
        
        # 选择 map 函数 (batched or not)
        map_fn = self._map_function_to_count_chunks_batched if use_batched_map else self._map_function_to_count_chunks
        
        # 执行 map 操作
        # 这个操作会返回一个新的 Dataset 对象（或者如果原始数据集是内存映射的，可能会就地修改）
        # 它会包含一个名为 "num_chunks_for_doc" 的新列。

        dataset_with_counts = self.hf_dataset.map(
            map_fn,
            batched=use_batched_map,
            batch_size=map_batch_size if use_batched_map else None,
            num_proc=num_proc_for_counting,
            desc="Counting chunks per document",
            remove_columns=[col for col in self.hf_dataset.column_names if col != self.text_column and col != "num_chunks_for_doc"] # 移除不必要的列以节省内存
        )

        # 从新列中计算总和
        # dataset_with_counts["num_chunks_for_doc"] 会是一个包含每个文档chunk数的列表或Arrow Array
        total_count = sum(dataset_with_counts["num_chunks_for_doc"])

        self._total_samples = total_count
        return self._total_samples
    
    def __len__(self):
        """
        返回数据集中的样本总数。
        如果未预计算，则会触发计算或抛出异常。
        DataLoader需要这个方法来知道总共有多少数据，以便正确显示进度。
        """
        if self._total_samples is None:
            raise RuntimeError("Total number of samples has not been precomputed. "
                               "Initialize dataset with precompute_total_samples=True "
                               "or call _compute_total_samples() manually.")
        return self._total_samples
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            doc_indices_for_this_worker = range(len(self.hf_dataset))
        else:
            num_docs = len(self.hf_dataset)
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            doc_indices_for_this_worker = range(worker_id, num_docs, num_workers)

        for doc_idx in doc_indices_for_this_worker:
            try:
                document_text = self.hf_dataset[doc_idx][self.text_column]
            except IndexError:
                print(f"Warning: Worker {worker_info.id if worker_info else 'main'} tried to access out-of-bounds doc_idx {doc_idx}.")
                continue
            for sample in self._tokenize_and_chunk_document_stream(document_text):
                yield sample


def custom_collate_fn(batch, tokenizer, max_seq_length):
    """
    自定义的collate_fn，使用tokenizer的pad功能
    """
    # 提取input_ids列表
    input_ids_list = [item['input_ids'] for item in batch]
    
    # 使用tokenizer的padding功能
    padded_batch = tokenizer.pad(
        {'input_ids': input_ids_list},
        padding='longest',  # 填充到batch中最长序列的长度
        max_length=max_seq_length,  # 但不超过最大长度
        return_tensors='pt',  # 返回PyTorch张量
        return_attention_mask=True
    )

    return {
        "input_ids": padded_batch['input_ids'],
        "attention_mask": padded_batch['attention_mask']
    }

