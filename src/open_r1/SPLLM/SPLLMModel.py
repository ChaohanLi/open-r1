import os
from typing import Optional, Dict, List, Tuple
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from .SPLLMProcessor import SPLLMProcessor

class SPLLMConfig(PretrainedConfig):
    model_type = "spllm"
    def __init__(
        self,
        text_model_name: str = "Qwen/Qwen3-1.7B",
        protein_model_name: str = "facebook/esm2_t12_35M_UR50D",
        text_model_finetune: bool = True,
        protein_model_finetune: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.text_model_name = text_model_name
        self.protein_model_name = protein_model_name
        self.text_model_finetune = text_model_finetune
        self.protein_model_finetune = protein_model_finetune
        self.name_or_path = text_model_name

class SPLLMModel(PreTrainedModel):
    config_class = SPLLMConfig

    def __init__(self, config: SPLLMConfig, cache_dir: Optional[str] = None, **kwargs):
        super().__init__(config)

        self.name_or_path = config.text_model_name
        self._name_or_path = config.text_model_name
        self.config.name_or_path = config.text_model_name

        self.text_model = AutoModelForCausalLM.from_pretrained(
            config.text_model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="eager"
        )
        self.text_tokenizer = AutoTokenizer.from_pretrained(config.text_model_name, trust_remote_code=True)
        self.protein_model = AutoModelForMaskedLM.from_pretrained(config.protein_model_name)
        self.protein_tokenizer = AutoTokenizer.from_pretrained(config.protein_model_name)

        # 占位符特殊 token
        self.protein_token = "<|protein_pad|>"
        added = self.text_tokenizer.add_special_tokens({"additional_special_tokens": [self.protein_token]})
        if added > 0:
            self.text_model.resize_token_embeddings(len(self.text_tokenizer))
        if self.text_tokenizer.pad_token is None:
            self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
        self.protein_token_id = self.text_tokenizer.convert_tokens_to_ids(self.protein_token)

        # 投影层
        prot_hidden = self._infer_protein_hidden()

        self.protein_projection = nn.Linear(prot_hidden, self.text_model.config.hidden_size)
        nn.init.xavier_uniform_(self.protein_projection.weight)
        if self.protein_projection.bias is not None:
            nn.init.zeros_(self.protein_projection.bias)

        # 冻结策略
        for p in self.text_model.parameters():
            p.requires_grad = config.text_model_finetune  # True
        for p in self.protein_model.parameters():
            p.requires_grad = False                       # 强制冻结 ESM
        self.protein_model.eval()                         # 推理模式（禁 dropout）
        for p in self.protein_projection.parameters():
            p.requires_grad = True

        # 处理器
        self.processor = SPLLMProcessor(
            tokenizer=self.text_tokenizer,
            protein_tokenizer=self.protein_tokenizer,
            protein_token=self.protein_token,
        )

    def _infer_protein_hidden(self) -> int:
        # 1) 直接读 config
        if hasattr(self.protein_model, "config") and hasattr(self.protein_model.config, "hidden_size"):
            return int(self.protein_model.config.hidden_size)
        # 2) base_model 的 config
        base = getattr(self.protein_model, "base_model", None)
        if base is not None and hasattr(base, "config") and hasattr(base.config, "hidden_size"):
            return int(base.config.hidden_size)
        # 3) 从嵌入矩阵推断（通常等于 hidden_size）
        emb = self.protein_model.get_input_embeddings() if hasattr(self.protein_model, "get_input_embeddings") else None
        if emb is not None and hasattr(emb, "weight"):
            return int(emb.weight.shape[1])
        # 4) 明确报错，避免静默用错维度
        raise ValueError("Cannot infer protein model hidden_size; please set it explicitly.")

    def encode_protein(self, prot_tok: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 冻结 ESM：no_grad 提速省显存；梯度仍会更新 projection
        with torch.no_grad():
            if hasattr(self.protein_model, "base_model"):
                out = self.protein_model.base_model(
                    input_ids=prot_tok["input_ids"],
                    attention_mask=prot_tok["attention_mask"],
                    return_dict=True,
                ).last_hidden_state
            else:
                out = self.protein_model(
                    input_ids=prot_tok["input_ids"],
                    attention_mask=prot_tok["attention_mask"],
                    return_dict=True,
                ).last_hidden_state
        if out.size(1) > 2:
            out = out[:, 1:-1, :]
        out = out.to(self.protein_projection.weight.dtype).to(self.protein_projection.weight.device)
        return self.protein_projection(out)  # projection 保持可训练

    def fuse_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, prot_emb: torch.Tensor):
        text_embeds = self.text_model.get_input_embeddings()(input_ids)
        text_embeds = text_embeds.to(dtype=prot_emb.dtype)
        mask = (input_ids == self.protein_token_id)  # [B,T]
        n_slots = int(mask.sum().item())
        flat_prot = prot_emb.reshape(-1, prot_emb.size(-1))
        if n_slots != flat_prot.size(0):
            raise ValueError(f"Protein tokens and embeddings mismatch: {n_slots} vs {flat_prot.size(0)}")
        text_embeds[mask] = flat_prot
        return text_embeds, attention_mask

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: Optional[torch.Tensor] = None,
        protein_tokenized: Optional[Dict[str, torch.Tensor]] = None,
        batch_idx_map: Optional[List[int]] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if protein_tokenized is not None and batch_idx_map is not None:
            prot_emb = self.encode_protein(protein_tokenized)
            fused, am = self.fuse_embeddings(input_ids, attention_mask, prot_emb)
            return self.text_model(inputs_embeds=fused, attention_mask=am, labels=labels, **kwargs)
        return self.text_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """启用 gradient checkpointing（委托给 text_model）"""
        if hasattr(self.text_model, 'gradient_checkpointing_enable'):
            self.text_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
    
    def gradient_checkpointing_disable(self):
        """禁用 gradient checkpointing"""
        if hasattr(self.text_model, 'gradient_checkpointing_disable'):
            self.text_model.gradient_checkpointing_disable()
    
    @property
    def supports_gradient_checkpointing(self):
        """声明支持 gradient checkpointing"""
        return hasattr(self.text_model, 'gradient_checkpointing_enable')