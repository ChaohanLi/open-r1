"""
Open-R1框架集成模块
提供标准接口，便于集成到Open-R1的SFT和GRPO流程中
"""

from typing import Optional, Dict, Any
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset

from .ProteinLLMModel import ProteinLLMModel, ProteinLLMConfig
from .ProteinDataCollator import ProteinLLMDataCollator, ProteinLLMDataCollatorForInference

def get_protein_llm_model(
    text_model_name: str = "Qwen/Qwen2.5-Math-7B",
    protein_model_name: str = "facebook/esm2_t33_650M_UR50D",
    **kwargs
) -> ProteinLLMModel:
    """
    工厂函数：创建蛋白质LLM模型
    兼容Open-R1的get_model接口
    """
    config = ProteinLLMConfig(
        text_model_name=text_model_name,
        protein_model_name=protein_model_name,
        **kwargs
    )
    return ProteinLLMModel(config=config)

def get_protein_data_collator(
    model: ProteinLLMModel,
    max_length_text: int = 512,
    max_length_protein: int = 100,
    for_inference: bool = False
) -> ProteinLLMDataCollator:
    """
    工厂函数：创建蛋白质数据收集器
    兼容Open-R1的数据处理流程
    """
    if for_inference:
        return ProteinLLMDataCollatorForInference(
            processor=model.processor,
            tokenizer=model.text_tokenizer,
            max_length_text=max_length_text,
            max_length_protein=max_length_protein
        )
    else:
        return ProteinLLMDataCollator(
            processor=model.processor,
            tokenizer=model.text_tokenizer,
            max_length_text=max_length_text,
            max_length_protein=max_length_protein
        )

def prepare_protein_dataset(dataset: Dataset) -> Dataset:
    """
    准备蛋白质数据集，确保格式兼容
    """
    def check_format(example):
        # 验证数据格式
        if "messages" not in example:
            raise ValueError("Dataset must contain 'messages' field")
        if "protein_sequence" not in example:
            raise ValueError("Dataset must contain 'protein_sequence' field")
        return example
    
    return dataset.map(check_format)

# 🔧 为Open-R1提供的标准接口
class ProteinLLMForOpenR1:
    """
    Open-R1框架适配器
    提供标准接口供SFT和GRPO使用
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        self.model = get_protein_llm_model(**model_config)
        self.data_collator = get_protein_data_collator(self.model)
    
    def get_model(self) -> PreTrainedModel:
        return self.model
    
    def get_tokenizer(self) -> PreTrainedTokenizer:
        return self.model.text_tokenizer
    
    def get_data_collator(self, for_inference: bool = False):
        return get_protein_data_collator(self.model, for_inference=for_inference)
    
    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        return prepare_protein_dataset(dataset)