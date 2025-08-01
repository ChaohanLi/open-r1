from typing import List, Optional, Union, Dict, Any
import torch
from transformers import ProcessorMixin, BatchFeature
from transformers.tokenization_utils_base import TextInput


class ProteinLLMProcessor(ProcessorMixin):
    """
    蛋白质多模态处理器 - 遵循HuggingFace设计模式
    
    职责：
    1. 使用两种tokenizer分别处理双模态数据
    2. 计算占位符数量并在文本中挖坑
    3. 返回处理后的单样本结果
    
    注意：不负责batch打包，那是DataCollator的职责
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
    
    # 🔧 新增：TRL兼容性属性
    @property
    def pad_token(self):
        """TRL期望的pad_token属性"""
        return self.tokenizer.pad_token
    
    @property
    def eos_token(self):
        """TRL期望的eos_token属性"""
        return self.tokenizer.eos_token
    
    @property
    def pad_token_id(self):
        """TRL期望的pad_token_id属性"""
        return self.tokenizer.pad_token_id
    
    @property
    def eos_token_id(self):
        """TRL期望的eos_token_id属性"""
        return self.tokenizer.eos_token_id

    @property
    def bos_token(self):
        """TRL期望的bos_token属性"""
        return getattr(self.tokenizer, 'bos_token', None)
    
    @property
    def bos_token_id(self):
        """TRL期望的bos_token_id属性"""
        return getattr(self.tokenizer, 'bos_token_id', None)
    
    # 🔧 TRL期望的方法 - 委托给内部tokenizer
    def apply_chat_template(self, *args, **kwargs):
        """委托给内部tokenizer"""
        return self.tokenizer.apply_chat_template(*args, **kwargs)
    
    def convert_tokens_to_ids(self, *args, **kwargs):
        """委托给内部tokenizer - TRL需要此方法"""
        return self.tokenizer.convert_tokens_to_ids(*args, **kwargs)
    
    def convert_ids_to_tokens(self, *args, **kwargs):
        """委托给内部tokenizer"""
        return self.tokenizer.convert_ids_to_tokens(*args, **kwargs)
    
    def encode(self, *args, **kwargs):
        """委托给内部tokenizer"""
        return self.tokenizer.encode(*args, **kwargs)
    
    def __len__(self):
        """返回tokenizer的词汇表大小"""
        return len(self.tokenizer)

    def __call__(
        self,
        text: Union[str, List[str]] = None,
        protein_sequence: Union[str, List[str]] = None,
        max_length_text: int = 1024,
        max_length_protein: int = 128,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        处理双模态数据 - 蛋白质序列是必需的
        
        Args:
            text: 文本数据（可以是字符串或字符串列表）
            protein_sequence: 蛋白质序列（字符串或字符串列表）- 必需
            max_length_text: 文本最大长度
            max_length_protein: 蛋白质最大长度
            return_tensors: 返回张量格式
            
        Returns:
            BatchFeature: 包含处理后的双模态数据
        """
        
        # 严格双模态要求
        if text is None or protein_sequence is None:
            raise ValueError("Both text and protein_sequence must be provided for dual-modal processing")
        
        # 1. 标准化输入格式
        if isinstance(text, str):
            text = [text]
        
        if isinstance(protein_sequence, str):
            protein_sequence = [protein_sequence]
        
        if len(text) != len(protein_sequence):
            raise ValueError(f"Text and protein sequence counts must match: {len(text)} vs {len(protein_sequence)}")
        
        batch_size = len(text)
        
        # 2. 处理蛋白质序列（ESM tokenizer接受字符串列表）
        print(f"Processing {len(protein_sequence)} protein sequences")
        
        protein_tokenized = self.protein_tokenizer(
            protein_sequence,  # 直接传入字符串列表
            padding=True,
            truncation=True,
            max_length=max_length_protein,
            return_tensors=return_tensors,
            return_attention_mask=True,
        )
        
        # 3. 为每个文本样本挖坑
        processed_texts = []
        for i, txt in enumerate(text):
            if self.protein_token in txt:
                # 计算该样本的实际氨基酸token数
                attention_mask = protein_tokenized['attention_mask'][i]
                total_tokens = attention_mask.sum().item()
                # ESM格式：[<cls>, AA1, AA2, ..., AAn, <eos>] → 氨基酸数 = total - 2
                amino_acid_count = max(1, total_tokens - 2)
                
                # 挖坑：将1个占位符扩展为对应数量的占位符
                expanded_text = txt.replace(
                    self.protein_token, 
                    self.protein_token * amino_acid_count
                )
                processed_texts.append(expanded_text)
                
                print(f"Sample {i}: {total_tokens} total tokens → {amino_acid_count} amino acid placeholders")
            else:
                processed_texts.append(txt)
        
        # 4. 处理文本（使用挖坑后的文本）
        text_tokenized = self.tokenizer(
            processed_texts,
            padding=True,
            truncation=True,
            max_length=max_length_text,
            return_tensors=return_tensors,
            **kwargs
        )
        
        # 5. 组装结果
        result = {
            # 文本tokenization结果
            **text_tokenized,
            # 蛋白质tokenization结果
            "protein_tokenized": {
                "input_ids": protein_tokenized["input_ids"],
                "attention_mask": protein_tokenized["attention_mask"],
            },
            # 简化的批次映射
            "batch_idx_map": list(range(batch_size)),
        }
        
        print(f"Processor output - Text: {text_tokenized['input_ids'].shape}, Protein: {protein_tokenized['input_ids'].shape}")
        
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