from ..SPLLM.SPLLMModel import SPLLMModel, SPLLMConfig
from ..SPLLM.SPLLMDataCollator import SPLLMDataCollator

def get_spllm_model(
    text_model_name: str = "Qwen/Qwen3-1.7B",
    protein_model_name: str = "facebook/esm2_t12_35M_UR50D",
    text_model_finetune: bool = True,
    protein_model_finetune: bool = False,
    **kwargs,
) -> SPLLMModel:
    cfg = SPLLMConfig(
        text_model_name=text_model_name,
        protein_model_name=protein_model_name,
        text_model_finetune=text_model_finetune,
        protein_model_finetune=protein_model_finetune,
    )
    return SPLLMModel(config=cfg, **kwargs)

def get_spllm_data_collator(
    model: SPLLMModel,
    max_length_text: int = 512,
    max_length_protein: int = 70,
    debug: bool = False,
) -> SPLLMDataCollator:
    return SPLLMDataCollator(
        processor=model.processor,
        max_length_text=max_length_text,
        max_length_protein=max_length_protein,
        debug=debug,
    )