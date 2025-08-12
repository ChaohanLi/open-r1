"""
蛋白质双模态模型工厂函数
与 Open-R1 框架集成
"""
from typing import Optional
from transformers import PreTrainedModel
from ..ProteinLLM.ProteinLLMModel import ProteinLLMModel, ProteinLLMConfig
from ..ProteinLLM.ProteinLLMDataCollator import ProteinLLMDataCollator

def get_protein_llm_model(
    text_model_name: str = "Qwen/Qwen3-1.7B", 
    protein_model_name: str = "facebook/esm2_t12_35M_UR50D",
    text_model_finetune: bool = True,
    protein_model_finetune: bool = False,
    **kwargs
) -> ProteinLLMModel:
    """
    创建蛋白质双模态模型
    
    与 Open-R1 的 get_model 接口兼容
    """
    config = ProteinLLMConfig(
        text_model_name=text_model_name,
        protein_model_name=protein_model_name, 
        text_model_finetune=text_model_finetune,
        protein_model_finetune=protein_model_finetune,
    )
    
    return ProteinLLMModel(config=config, **kwargs)

def get_protein_data_collator(
    model: ProteinLLMModel,
    max_length_text: int = 512,
    max_length_protein: int = 100,
    **kwargs
):
    """
    创建蛋白质数据收集器
    
    与 SFTTrainer 兼容
    """
    return ProteinLLMDataCollator(
        processor=model.processor,
        max_length_text=max_length_text,
        max_length_protein=max_length_protein,
        **kwargs
    )