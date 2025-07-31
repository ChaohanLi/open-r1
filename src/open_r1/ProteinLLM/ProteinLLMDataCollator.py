from typing import List, Dict, Any, Optional
import torch
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset

class ProteinLLMDataCollator:
    """
    蛋白质多模态数据收集器 - 遵循HuggingFace设计模式
    
    职责：
    1. 从原始features提取数据
    2. 调用Processor进行双模态处理
    3. 将处理结果打包成训练batch
    4. 添加训练标签
    
    分工：
    - Processor: 负责tokenization和挖坑
    - DataCollator: 负责batch打包和标签生成
    """
    
    def __init__(self, processor, max_length_text: int = 512, max_length_protein: int = 100):
        self.processor = processor
        self.max_length_text = max_length_text
        self.max_length_protein = max_length_protein
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        将features转换为训练batch
        
        Args:
            features: 原始数据样本列表，每个包含messages和protein_sequence
            
        Returns:
            Dict[str, torch.Tensor]: 训练batch
        """
        print(f"DataCollator received {len(features)} features")
        
        # 1. 提取数据
        texts = []
        protein_sequences = []
        
        for i, feature in enumerate(features):
            # 提取文本
            if "text" in feature:
                text = feature["text"]
            elif "messages" in feature:
                text = self.processor.tokenizer.apply_chat_template(
                    feature["messages"], 
                    tokenize=False, 
                    add_generation_prompt=False
                )
            else:
                raise ValueError(f"Feature {i} missing 'text' or 'messages' field")
            
            # 提取蛋白质序列（现在是字符串格式）
            if "protein_sequence" in feature:
                protein_seq = feature["protein_sequence"]
                
                # 验证格式和长度
                if not isinstance(protein_seq, str):
                    raise ValueError(f"Expected protein_sequence to be string, got {type(protein_seq)}")
                
                if len(protein_seq) != 70:
                    raise ValueError(f"Expected 70 amino acids, got {len(protein_seq)}")
                
                protein_sequences.append(protein_seq)
            else:
                raise ValueError(f"Feature {i} missing required 'protein_sequence' field")
            
            texts.append(text)
        
        # 2. 使用Processor处理双模态数据
        print(f"Using Processor to handle {len(texts)} samples")
        
        try:
            batch = self.processor(
                text=texts,
                protein_sequence=protein_sequences,
                max_length_text=self.max_length_text,
                max_length_protein=self.max_length_protein,
                return_tensors="pt"
            )
        except Exception as e:
            print(f"Processor error: {e}")
            # 降级处理：仅处理文本
            batch = self.processor.tokenizer(
                texts,
                max_length=self.max_length_text,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            batch["protein_tokenized"] = None
            batch["batch_idx_map"] = []
        
        # 3. 添加训练标签
        batch["labels"] = batch["input_ids"].clone()
        
        # 4. 确保batch是字典格式
        if not isinstance(batch, dict):
            batch = dict(batch)
        
        print(f"DataCollator output keys: {batch.keys()}")
        if batch.get("input_ids") is not None:
            print(f"Batch input_ids shape: {batch['input_ids'].shape}")
        if batch.get("protein_tokenized") is not None:
            print(f"Batch protein shape: {batch['protein_tokenized']['input_ids'].shape}")
        
        return batch

class ProteinLLMDataCollatorForInference(ProteinLLMDataCollator):
    """推理专用数据收集器（不生成labels）"""
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(features)
        # 移除labels，保留其他字段用于推理
        if "labels" in batch:
            del batch["labels"]
        return batch