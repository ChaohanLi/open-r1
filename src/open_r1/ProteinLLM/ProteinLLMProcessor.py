from typing import List, Optional, Union, Dict, Any
import torch
from transformers import ProcessorMixin, BatchFeature
from transformers.tokenization_utils_base import TextInput


class ProteinLLMProcessor(ProcessorMixin):
    """
    简化的蛋白质处理器 - 专门针对信号肽分类任务
    假设：每个样本确定包含1个长度70的蛋白质序列
    """
    
    attributes = ["tokenizer", "protein_tokenizer"]
    valid_kwargs = ["model"]
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")
    protein_tokenizer_class = ("EsmTokenizer",)
    
    def __init__(self, tokenizer=None, protein_tokenizer=None, **kwargs):
        self.tokenizer = tokenizer
        self.protein_tokenizer = protein_tokenizer
        self.protein_token = "<|protein_pad|>"
        
        super().__init__(tokenizer, protein_tokenizer)
        
        # 确保pad_token存在
        if not hasattr(self.tokenizer, 'pad_token') or self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __call__(
        self,
        batch_protein_sequences: List[List[str]],
        text: Union[str, List[str]],
        max_length_text: int = 1024,
        max_length_protein: int = 128,
        return_tensors: str = "pt",
        **kwargs,
    ) -> BatchFeature:
        """
        简化的处理方法 - 修复ESM token计算
        """
        
        # 确保text是列表
        if isinstance(text, str):
            text = [text]
        
        batch_size = len(text)
        
        # 简化：直接提取蛋白质序列（每个样本1个）
        protein_sequences = []
        for sequences in batch_protein_sequences:
            assert len(sequences) == 1, f"Expected 1 protein per sample, got {len(sequences)}"
            protein_sequences.append(sequences[0])
        
        # 处理蛋白质序列
        if protein_sequences:
            protein_tokenized = self.protein_tokenizer(
                protein_sequences,
                padding=True,
                truncation=True,
                max_length=max_length_protein,
                return_tensors=return_tensors,
                return_attention_mask=True,
            )
            
            # 简化的占位符替换：一对一映射
            processed_text = []
            for i, txt in enumerate(text):
                if self.protein_token in txt:
                    # 🔧 修复：正确计算蛋白质token数量（排除<cls>和<eos>）
                    attention_mask = protein_tokenized['attention_mask'][i]
                    total_tokens = attention_mask.sum().item()
                    
                    # ESM会添加<cls>和<eos>，所以实际蛋白质token数 = total - 2
                    protein_token_count = max(1, total_tokens - 2)  # 至少保留1个token
                    
                    # 简单替换：一个占位符变成多个
                    processed_txt = txt.replace(
                        self.protein_token, 
                        self.protein_token * protein_token_count
                    )
                    processed_text.append(processed_txt)
                else:
                    processed_text.append(txt)
            
            text = processed_text
        else:
            protein_tokenized = None
        
        # 处理文本
        text_inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length_text,
            return_tensors=return_tensors,
            **kwargs
        )
        
        # 组装结果
        result = {**text_inputs}
        if protein_tokenized is not None:
            result["protein_tokenized"] = protein_tokenized
            # 简化的batch_idx_map：[0, 1, 2, ...]
            result["batch_idx_map"] = list(range(batch_size))
        
        return BatchFeature(data=result)
    
    def batch_decode(self, *args, **kwargs) -> List[str]:
        return self.tokenizer.batch_decode(*args, **kwargs)
    
    def decode(self, *args, **kwargs) -> str:
        return self.tokenizer.decode(*args, **kwargs)
    
    @property
    def model_input_names(self) -> List[str]:
        tokenizer_names = self.tokenizer.model_input_names
        protein_names = ["protein_tokenized", "batch_idx_map"]
        return list(dict.fromkeys(tokenizer_names + protein_names))