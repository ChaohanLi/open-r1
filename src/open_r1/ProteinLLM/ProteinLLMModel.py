from typing import Optional, Dict, Any, List, Union, Tuple
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForMaskedLM,
    PreTrainedModel,
    PretrainedConfig
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from .ProteinLLMProcessor import ProteinLLMProcessor


class ProteinLLMModel(PreTrainedModel):
    """
    蛋白质信号肽分类模型
    结合文本LLM和蛋白质编码器，专门用于信号肽类型识别任务
    
    Architecture:
        - Text Model: Qwen3 (处理对话和推理)
        - Protein Model: ESM2 (编码蛋白质序列)
        - Projection: 将蛋白质embedding投影到文本空间
        - Fusion: 在文本中替换<|protein_pad|>占位符
    """
    
    def __init__(
        self,
        config,
        text_model_name: str = "Qwen/Qwen3-1.7B",
        protein_model_name: str = "facebook/esm2_t33_650M_UR50D",
        cache_dir: Optional[str] = None,
        text_model_finetune: bool = True,
        protein_model_finetune: bool = False,  # 冻结蛋白质模型（Cold Start策略）
        **kwargs
    ):
        super().__init__(config)
        
        self.text_model_name = text_model_name
        self.protein_model_name = protein_model_name
        self.text_model_finetune = text_model_finetune
        self.protein_model_finetune = protein_model_finetune

        self.name_or_path = config.text_model_name
        self.config.name_or_path = config.text_model_name
        
        # 加载文本模型
        print(f"Loading text model: {text_model_name}")
        self.text_model = AutoModelForCausalLM.from_pretrained(
            text_model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            # 🔧 使用bfloat16平衡精度和显存，数值稳定性好
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            **kwargs
        )
        
        # 加载文本tokenizer并配置特殊token
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            text_model_name,
            trust_remote_code=True
        )
        
        self.config.tokenizer = self.text_tokenizer

        # 添加蛋白质特殊token（Model的职责）
        protein_tokens = ["<|protein_pad|>"]
        num_added = self.text_tokenizer.add_special_tokens({
            "additional_special_tokens": protein_tokens
        })
        
        if num_added > 0:
            # 扩展文本模型的embedding
            self.text_model.resize_token_embeddings(len(self.text_tokenizer))
        
        # 获取特殊token ID
        self.protein_token_id = self.text_tokenizer.convert_tokens_to_ids("<|protein_pad|>")
        
        # 确保pad_token存在
        if self.text_tokenizer.pad_token is None:
            self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
        
        # 加载蛋白质模型
        print(f"Loading protein model: {protein_model_name}")
        self.protein_model = AutoModelForMaskedLM.from_pretrained(
            protein_model_name,
            cache_dir=cache_dir
        )
        
        # 加载蛋白质tokenizer
        self.protein_tokenizer = AutoTokenizer.from_pretrained(protein_model_name)
        
        # 创建投影层：蛋白质embedding → 文本embedding空间
        self.protein_projection = nn.Linear(
            self.protein_model.config.hidden_size,  # ESM2: 1280
            self.text_model.config.hidden_size      # Qwen3-1.7B: 2048
        )
        
        # 🔧 参考BioReason：添加投影层权重初始化，提高数值稳定性
        with torch.no_grad():
            # 使用Xavier uniform初始化，减少数值不稳定
            nn.init.xavier_uniform_(self.protein_projection.weight)
            # bias初始化为0
            if self.protein_projection.bias is not None:
                nn.init.zeros_(self.protein_projection.bias)
        
        # 设置模型的可训练性
        self._set_model_trainability()
        
        # 创建处理器（使用已配置的tokenizer）
        self.processor = ProteinLLMProcessor(
            tokenizer=self.text_tokenizer,
            protein_tokenizer=self.protein_tokenizer
        )
        
        print(f"ProteinLLMModel initialized successfully!")
        print(f"Text vocab size: {len(self.text_tokenizer)}")
        print(f"Protein token ID: {self.protein_token_id}")
    
    def _set_model_trainability(self):
        """设置模型的可训练性（Cold Start策略）"""
        
        # 文本模型的可训练性
        for param in self.text_model.parameters():
            param.requires_grad = self.text_model_finetune
        
        # 蛋白质模型的可训练性（通常冻结以节省计算）
        for param in self.protein_model.parameters():
            param.requires_grad = self.protein_model_finetune
        
        # 投影层始终可训练
        for param in self.protein_projection.parameters():
            param.requires_grad = True
        
        print(f"Text model trainable: {self.text_model_finetune}")
        print(f"Protein model trainable: {self.protein_model_finetune}")
        print(f"Projection layer trainable: True")
    
    def encode_protein_sequences(
        self, 
        protein_tokenized: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        使用ESM2获取逐氨基酸表示 - 修正版本
        
        ESM2结构：
        - EsmForMaskedLM.esm (基础编码器)
        - EsmForMaskedLM.esm.embeddings
        - EsmForMaskedLM.esm.encoder
        - EsmForMaskedLM.lm_head (MLM头，我们不需要)
        """
        with torch.set_grad_enabled(self.protein_model_finetune):
            # 🔧 修正：使用ESM2的基础编码器获取逐氨基酸表示
            # 直接调用esm编码器，避免MaskedLM的复杂性
            esm_base_model = self.protein_model.esm  # 获取基础ESM编码器
            
            # 获取逐氨基酸的表示
            protein_outputs = esm_base_model(
                input_ids=protein_tokenized["input_ids"],
                attention_mask=protein_tokenized["attention_mask"],
                return_dict=True
            )
            
            # 现在可以获取last_hidden_state了
            protein_embeddings = protein_outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
            
            # 🔧 移除ESM的特殊tokens：<cls> (pos 0) 和 <eos> (pos -1)
            # ESM tokenizer格式：[<cls>, AA1, AA2, ..., AAn, <eos>]
            # 我们只要中间的氨基酸表示：[AA1, AA2, ..., AAn]
            if protein_embeddings.size(1) > 2:  # 确保序列长度足够
                protein_embeddings = protein_embeddings[:, 1:-1, :]  # 移除首尾特殊tokens
            
            # 🔧 参考BioReason：确保设备和数据类型一致
            protein_embeddings = protein_embeddings.to(
                device=self.protein_projection.weight.device,
                dtype=self.protein_projection.weight.dtype
            )
            
            # 应用投影层：ESM hidden_size -> Text hidden_size  
            protein_embeddings = self.protein_projection(protein_embeddings)
            
            return protein_embeddings

    def fuse_protein_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        protein_embeddings: torch.Tensor,
        batch_idx_map: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        🔧 重写：参考BioReason的高效融合方法
        使用布尔掩码批量替换，避免循环引入的不稳定性
        """
        # 获取文本embeddings
        text_embeddings = self.text_model.get_input_embeddings()(input_ids)
        
        # 🔧 参考BioReason：确保protein_embeddings与text_embeddings数据类型一致
        if protein_embeddings is not None:
            protein_embeddings = protein_embeddings.to(dtype=text_embeddings.dtype)
        
            # 找到所有蛋白质占位符位置
            mask = (input_ids == self.protein_token_id)
            n_protein_tokens = mask.sum().item()
            
            # 扁平化所有蛋白质embeddings
            protein_embeds_flat = protein_embeddings.view(-1, protein_embeddings.size(-1))
            n_protein_features = protein_embeds_flat.shape[0]
            
            # 🔧 严格检查：确保数量匹配
            if n_protein_features != n_protein_tokens:
                raise ValueError(
                    f"Protein features and protein tokens do not match: "
                    f"features {n_protein_features}, tokens: {n_protein_tokens}"
                )
            
            # 🔧 参考BioReason：使用布尔掩码批量替换
            text_embeddings[mask] = protein_embeds_flat
        
        return text_embeddings, attention_mask
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        protein_tokenized: Optional[Dict[str, torch.Tensor]] = None,
        batch_idx_map: Optional[List[int]] = None,
        **kwargs
    ) -> CausalLMOutputWithPast:
        """
        前向传播
        
        Args:
            input_ids: 文本token IDs
            attention_mask: 文本attention mask  
            labels: 训练标签
            protein_tokenized: 蛋白质tokenization结果
            batch_idx_map: 蛋白质到batch的映射
            
        Returns:
            CausalLMOutputWithPast: 包含loss和logits
        """
        
        # 1. 编码蛋白质序列（如果存在）
        protein_embeddings = None
        if protein_tokenized is not None:
            protein_embeddings = self.encode_protein_sequences(protein_tokenized)
        
        # 2. 融合蛋白质和文本表示
        if protein_embeddings is not None and batch_idx_map is not None:
            fused_embeddings, fused_attention_mask = self.fuse_protein_embeddings(
                input_ids=input_ids,
                attention_mask=attention_mask,
                protein_embeddings=protein_embeddings,
                batch_idx_map=batch_idx_map
            )
            
            # 🔧 添加数值稳定性检查
            if torch.isnan(fused_embeddings).any():
                raise ValueError("NaN detected in fused_embeddings before text_model forward")
            
            # 🔧 添加数值范围检查，防止极值
            if fused_embeddings.abs().max() > 1e6:
                print(f"Warning: Large values in fused_embeddings: max={fused_embeddings.abs().max()}")
            
            # 使用融合后的embeddings
            outputs = self.text_model(
                inputs_embeds=fused_embeddings,
                attention_mask=fused_attention_mask,
                labels=labels,
                **kwargs
            )
        else:
            # 纯文本模式
            outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )
        
        return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        protein_tokenized: Optional[Dict[str, torch.Tensor]] = None,
        batch_idx_map: Optional[List[int]] = None,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> torch.Tensor:
        """
        生成文本（用于推理）
        
        Args:
            input_ids: 输入token IDs
            attention_mask: attention mask
            protein_tokenized: 蛋白质数据
            batch_idx_map: 批次映射
            max_new_tokens: 最大生成token数
            **kwargs: 其他生成参数
            
        Returns:
            generated_ids: 生成的token IDs
        """
        
        # 处理蛋白质embeddings
        if protein_tokenized is not None and batch_idx_map is not None:
            protein_embeddings = self.encode_protein_sequences(protein_tokenized)
            fused_embeddings, fused_attention_mask = self.fuse_protein_embeddings(
                input_ids=input_ids,
                attention_mask=attention_mask,
                protein_embeddings=protein_embeddings,
                batch_idx_map=batch_idx_map
            )
            
            # 使用融合embeddings生成
            generated_ids = self.text_model.generate(
                inputs_embeds=fused_embeddings,
                attention_mask=fused_attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.text_tokenizer.pad_token_id,
                eos_token_id=self.text_tokenizer.eos_token_id,
                **kwargs
            )
        else:
            # 纯文本生成
            generated_ids = self.text_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.text_tokenizer.pad_token_id,
                eos_token_id=self.text_tokenizer.eos_token_id,
                **kwargs
            )
        
        return generated_ids
    
    # 🔧 添加梯度检查点支持方法
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """启用梯度检查点"""
        if hasattr(self.text_model, 'gradient_checkpointing_enable'):
            self.text_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
    
    def gradient_checkpointing_disable(self):
        """禁用梯度检查点"""
        if hasattr(self.text_model, 'gradient_checkpointing_disable'):
            self.text_model.gradient_checkpointing_disable()
    
    @property
    def supports_gradient_checkpointing(self):
        """检查是否支持梯度检查点"""
        return hasattr(self.text_model, 'gradient_checkpointing_enable')
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """从SFT检查点加载（本地优先）"""
        # 1) 读取配置
        config = ProteinLLMConfig.from_pretrained(pretrained_model_name_or_path)
        # 2) 文本模型子目录（优先本地）
        text_model_dir = os.path.join(pretrained_model_name_or_path, "text_model")
        text_model_name = text_model_dir if os.path.isdir(text_model_dir) else config.text_model_name
        # 3) 初始化（会从本地 text_model_dir 加载）
        model = cls(
            config=config,
            text_model_name=text_model_name,
            protein_model_name=config.protein_model_name,
            **kwargs
        )
        # 4) 加载投影层
        proj_path = os.path.join(pretrained_model_name_or_path, "protein_projection.pth")
        if os.path.exists(proj_path):
            state = torch.load(proj_path, map_location="cpu")
            model.protein_projection.load_state_dict(state)
        # 5) 覆盖 tokenizer/processor（若已保存）
        tok_dir = os.path.join(pretrained_model_name_or_path, "text_tokenizer")
        if os.path.isdir(tok_dir):
            model.text_tokenizer = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
        proc_dir = os.path.join(pretrained_model_name_or_path, "processor")
        if os.path.isdir(proc_dir):
            model.processor = ProteinLLMProcessor.from_pretrained(proc_dir)
        return model
    
    def save_pretrained(self, save_directory, **kwargs):
        """保存模型到本地检查点"""
        os.makedirs(save_directory, exist_ok=True)
        # 1) 保存config
        self.config.save_pretrained(save_directory)
        # 2) 保存文本模型/分词器
        self.text_model.save_pretrained(os.path.join(save_directory, "text_model"), **kwargs)
        self.text_tokenizer.save_pretrained(os.path.join(save_directory, "text_tokenizer"))
        # 3) 保存投影层
        torch.save(self.protein_projection.state_dict(), os.path.join(save_directory, "protein_projection.pth"))
        # 4) 保存处理器
        self.processor.save_pretrained(os.path.join(save_directory, "processor"), **kwargs)


# 用于配置的简单config类
class ProteinLLMConfig(PretrainedConfig):
    """ProteinLLM配置类"""
    
    model_type = "protein_llm"
    
    def __init__(
        self,
        text_model_name: str = "Qwen/Qwen3-1.7B",
        protein_model_name: str = "facebook/esm2_t33_650M_UR50D",
        text_model_finetune: bool = True,
        protein_model_finetune: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.text_model_name = text_model_name
        self.protein_model_name = protein_model_name
        self.text_model_finetune = text_model_finetune
        self.protein_model_finetune = protein_model_finetune