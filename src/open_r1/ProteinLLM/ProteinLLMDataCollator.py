from typing import List, Dict, Any, Optional
import torch
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset

class ProteinLLMDataCollator:
    """
    蛋白质多模态数据收集器
    
    职责：
    1. 接收TRL标准格式数据（包含text和protein_sequence）
    2. 使用ProteinLLMProcessor处理双模态数据
    3. 返回模型期望的完整batch格式
    
    工作流程：
    TRL数据 -> 提取蛋白质序列 -> Processor处理 -> 模型输入格式
    """
    
    def __init__(self, processor, tokenizer, max_length_text: int = 512, max_length_protein: int = 100):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length_text = max_length_text
        self.max_length_protein = max_length_protein
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        处理batch数据
        
        Args:
            features: TRL传入的特征列表，每个feature包含：
                - text: 格式化后的对话文本（TRL已处理）
                - protein_sequence: 蛋白质序列字符串（原始数据保留）
                
        Returns:
            batch: 模型期望的输入格式
                - input_ids: 文本token IDs
                - attention_mask: 文本attention mask
                - labels: 训练标签
                - protein_tokenized: 蛋白质tokenization结果
                - batch_idx_map: 蛋白质到batch的映射
        """
        
        # 🔧 步骤1：提取数据
        texts = []
        protein_sequences = []
        
        for feature in features:
            # 提取TRL格式化的文本
            if "text" in feature:
                texts.append(feature["text"])
            else:
                raise ValueError("DataCollator expects 'text' field from TRL processing")
            
            # 提取蛋白质序列
            if "protein_sequence" in feature:
                # 确保是字符串格式
                protein_seq = feature["protein_sequence"]
                if isinstance(protein_seq, list) and len(protein_seq) == 1:
                    protein_seq = protein_seq[0]
                elif isinstance(protein_seq, list):
                    raise ValueError(f"Expected 1 protein sequence per sample, got {len(protein_seq)}")
                protein_sequences.append(protein_seq)
            else:
                # 没有蛋白质序列，使用占位符
                protein_sequences.append("")
        
        # 🔧 步骤2：使用你的Processor处理双模态数据
        try:
            # 转换为Processor期望的格式
            batch_protein_sequences = [[seq] for seq in protein_sequences if seq]
            
            if batch_protein_sequences:
                # 有蛋白质序列的情况
                batch = self.processor(
                    batch_protein_sequences=batch_protein_sequences,
                    text=texts,
                    max_length_text=self.max_length_text,
                    max_length_protein=self.max_length_protein,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
            else:
                # 纯文本情况（备选方案）
                batch = self.processor.tokenizer(
                    texts,
                    max_length=self.max_length_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                # 添加空的蛋白质信息
                batch["protein_tokenized"] = None
                batch["batch_idx_map"] = []
                
        except Exception as e:
            print(f"Processor error: {e}")
            # 降级处理：纯文本
            batch = self.processor.tokenizer(
                texts,
                max_length=self.max_length_text,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            batch["protein_tokenized"] = None
            batch["batch_idx_map"] = []
        
        # 🔧 步骤3：添加训练标签
        batch["labels"] = batch["input_ids"].clone()
        
        # 🔧 步骤4：确保batch字典格式正确
        if not isinstance(batch, dict):
            batch = dict(batch)
        
        return batch

class ProteinLLMDataCollatorForInference(ProteinLLMDataCollator):
    """推理专用数据收集器（不生成labels）"""
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(features)
        # 移除labels，保留其他字段用于推理
        if "labels" in batch:
            del batch["labels"]
        return batch