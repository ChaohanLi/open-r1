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
        """处理batch数据"""
        print(f"DataCollator received {len(features)} features")
        
        # 提取数据
        texts = []
        protein_sequences = []
        
        for i, feature in enumerate(features):
            # 提取文本
            if "text" in feature:
                texts.append(feature["text"])
            elif "messages" in feature:
                text = self.tokenizer.apply_chat_template(
                    feature["messages"], 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                texts.append(text)
            else:
                raise ValueError(f"Feature {i} missing 'text' or 'messages' field")
            
            # 提取蛋白质序列
            if "protein_sequence" in feature:
                protein_seq = feature["protein_sequence"]
                if isinstance(protein_seq, list) and len(protein_seq) == 1:
                    protein_seq = protein_seq[0]
                elif isinstance(protein_seq, list):
                    raise ValueError(f"Expected 1 protein sequence per sample, got {len(protein_seq)}")
                protein_sequences.append(protein_seq)
            else:
                print(f"Warning: Feature {i} missing 'protein_sequence', using empty string")
                protein_sequences.append("")
        
        # 🔧 修复：使用Processor处理，避免参数冲突
        try:
            batch_protein_sequences = [[seq] for seq in protein_sequences if seq]
            
            if batch_protein_sequences:
                print(f"Processing {len(batch_protein_sequences)} protein sequences")
                # 🔧 关键修复：不传递额外的kwargs，避免参数冲突
                batch = self.processor(
                    batch_protein_sequences=batch_protein_sequences,
                    text=texts,
                    max_length_text=self.max_length_text,
                    max_length_protein=self.max_length_protein,
                    return_tensors="pt"
                    # 🔧 移除padding和truncation参数，让processor内部处理
                )
            else:
                print("No protein sequences found, using text-only processing")
                # 🔧 纯文本处理也要避免参数冲突
                batch = self.tokenizer(
                    texts,
                    max_length=self.max_length_text,
                    return_tensors="pt",
                    padding=True,  # 只在这里指定一次
                    truncation=True
                )
                batch["protein_tokenized"] = None
                batch["batch_idx_map"] = []
                
        except Exception as e:
            print(f"Processor error: {e}")
            print("Falling back to text-only processing")
            # 🔧 降级处理
            batch = self.tokenizer(
                texts,
                max_length=self.max_length_text,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            batch["protein_tokenized"] = None
            batch["batch_idx_map"] = []
        
        # 添加训练标签
        batch["labels"] = batch["input_ids"].clone()
        
        # 确保batch是字典格式
        if not isinstance(batch, dict):
            batch = dict(batch)
        
        print(f"DataCollator output keys: {batch.keys()}")
        print(f"Batch input_ids shape: {batch['input_ids'].shape}")
        
        return batch

class ProteinLLMDataCollatorForInference(ProteinLLMDataCollator):
    """推理专用数据收集器（不生成labels）"""
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(features)
        # 移除labels，保留其他字段用于推理
        if "labels" in batch:
            del batch["labels"]
        return batch