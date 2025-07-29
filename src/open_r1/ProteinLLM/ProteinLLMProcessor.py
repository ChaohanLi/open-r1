from typing import List, Optional, Union, Dict, Any
import torch
from transformers import (
    ProcessorMixin, 
    BatchFeature,
    AutoTokenizer
)
from transformers.tokenization_utils_base import TextInput


class ProteinLLMProcessor(ProcessorMixin):
    """
    蛋白质-文本双模态处理器
    专门用于处理包含蛋白质序列的对话数据
    参考BioReason的DLProcessor设计，适配信号肽分类任务
    """
    
    # ProcessorMixin要求的属性
    attributes = ["tokenizer", "protein_tokenizer"]
    valid_kwargs = ["model", "chat_template"]
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast", "LlamaTokenizer", "LlamaTokenizerFast")
    protein_tokenizer_class = ("EsmTokenizer",)
    
    def __init__(
        self, 
        tokenizer=None, 
        protein_tokenizer=None, 
        **kwargs
    ):
        """
        初始化蛋白质处理器
        
        Args:
            tokenizer: 文本tokenizer (如Qwen2Tokenizer)
            protein_tokenizer: 蛋白质tokenizer (如EsmTokenizer)
            chat_template: 聊天模板 (可选，默认使用tokenizer的模板)
        """
        self.tokenizer = tokenizer
        self.protein_tokenizer = protein_tokenizer
        
        # 蛋白质特殊token
        self.protein_token = "<|protein_pad|>"
        
        # 调用父类初始化
        super().__init__(tokenizer, protein_tokenizer)
        
        # 确保pad_token存在 (为GRPO训练准备)
        if not hasattr(self.tokenizer, 'pad_token') or self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 添加蛋白质特殊token到词汇表
        if self.protein_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({
                "additional_special_tokens": [self.protein_token]
            })
    
    def tokenize_protein_sequences(
        self, 
        batch_protein_sequences: List[List[str]], 
        max_length: int = 1024,
        return_tensors: str = "pt",
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """
        Tokenize蛋白质序列批次
        
        Args:
            batch_protein_sequences: 每个batch item的蛋白质序列列表 [[seq1], [seq2], ...]
            max_length: 蛋白质序列最大长度
            return_tensors: 返回张量格式
            device: 设备
            
        Returns:
            Dict包含:
                - protein_tokenized: tokenized的蛋白质序列
                - batch_idx_map: 序列到batch的映射关系
        """
        batch_idx_map = []
        all_sequences = []
        
        # 展平所有蛋白质序列并记录batch映射 (参考BioReason逻辑)
        for batch_idx, protein_sequences in enumerate(batch_protein_sequences):
            for seq in protein_sequences:
                all_sequences.append(seq)
                batch_idx_map.append(batch_idx)
        
        # 如果没有蛋白质序列，返回空结果
        if not all_sequences:
            return {"protein_tokenized": None, "batch_idx_map": []}
        
        # 一次性tokenize所有蛋白质序列
        protein_tokenized = self.protein_tokenizer(
            all_sequences,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors=return_tensors,
            return_attention_mask=True,
            add_special_tokens=True,
        )
        
        return {
            "protein_tokenized": protein_tokenized, 
            "batch_idx_map": batch_idx_map
        }
    
    def _replace_protein_placeholders(
        self, 
        text_list: List[str], 
        protein_result: Dict[str, Any]
    ) -> List[str]:
        """
        替换文本中的蛋白质占位符
        将单个<|protein_pad|>替换为多个token以匹配实际蛋白质序列长度
        (参考BioReason的DNA占位符替换逻辑)
        
        Args:
            text_list: 包含占位符的文本列表
            protein_result: tokenize_protein_sequences的返回结果
            
        Returns:
            处理后的文本列表
        """
        if protein_result["protein_tokenized"] is None:
            return text_list
        
        processed_text = []
        protein_idx = 0
        
        for batch_idx, text in enumerate(text_list):
            current_text = text
            
            # 处理当前文本中的所有蛋白质占位符
            while self.protein_token in current_text and protein_idx < len(protein_result["batch_idx_map"]):
                # 确认当前蛋白质序列属于当前batch
                if protein_result["batch_idx_map"][protein_idx] == batch_idx:
                    # 计算当前蛋白质序列的有效token数量
                    attention_mask = protein_result["protein_tokenized"]["attention_mask"][protein_idx]
                    num_protein_tokens = attention_mask.sum().item()
                    
                    # 使用临时占位符避免重复替换 (BioReason的策略)
                    current_text = current_text.replace(
                        self.protein_token, 
                        "<|placeholder|>" * num_protein_tokens, 
                        1  # 只替换第一个匹配
                    )
                    protein_idx += 1
                else:
                    break
            
            # 将临时占位符替换回蛋白质token
            current_text = current_text.replace("<|placeholder|>", self.protein_token)
            processed_text.append(current_text)
        
        return processed_text
    
    def __call__(
        self,
        batch_protein_sequences: Optional[List[List[str]]] = None,
        text: Optional[Union[TextInput, List[TextInput]]] = None,
        max_length_text: int = 1024,
        max_length_protein: int = 1024,
        return_tensors: str = "pt",
        device: str = "cuda",
        **kwargs,
    ) -> BatchFeature:
        """
        主要处理方法，SFTTrainer会调用此方法
        
        Args:
            batch_protein_sequences: 蛋白质序列批次 [[seq1], [seq2], ...]
            text: 包含占位符的文本 (SFTTrainer已经通过chat_template转换了messages)
            max_length_text: 文本最大长度
            max_length_protein: 蛋白质序列最大长度
            return_tensors: 返回张量格式
            device: 设备
            **kwargs: 其他tokenizer参数
            
        Returns:
            BatchFeature包含所有模型输入
        """
        # 确保text是列表格式
        if not isinstance(text, list):
            text = [text] if text is not None else []
        
        # 处理蛋白质序列
        protein_inputs = {}
        if batch_protein_sequences is not None:
            protein_processing_result = self.tokenize_protein_sequences(
                batch_protein_sequences,
                max_length=max_length_protein,
                return_tensors=return_tensors,
                device=device,
            )
            
            # 替换文本中的蛋白质占位符
            if text and protein_processing_result["protein_tokenized"] is not None:
                text = self._replace_protein_placeholders(text, protein_processing_result)
            
            # 准备蛋白质输入数据
            protein_inputs = {
                "protein_tokenized": protein_processing_result["protein_tokenized"],
                "batch_idx_map": protein_processing_result["batch_idx_map"],
            }
        
        # Tokenize处理后的文本
        text_kwargs = {k: v for k, v in kwargs.items() if k not in ['max_length_protein', 'device']}
        
        if text:
            text_inputs = self.tokenizer(
                text, 
                max_length=max_length_text + 2 * max_length_protein,  # 为蛋白质序列预留空间
                return_tensors=return_tensors,
                padding=True,
                truncation=True,
                **text_kwargs,
            )
        else:
            text_inputs = {}
        
        # 返回合并的BatchFeature
        return BatchFeature(data={**text_inputs, **protein_inputs})
    
    # ProcessorMixin要求的方法
    def batch_decode(self, *args, **kwargs) -> List[str]:
        """批量解码，转发给tokenizer"""
        return self.tokenizer.batch_decode(*args, **kwargs)
    
    def decode(self, *args, **kwargs) -> str:
        """解码，转发给tokenizer"""
        return self.tokenizer.decode(*args, **kwargs)
    
    @property
    def model_input_names(self) -> List[str]:
        """返回模型期望的输入字段名"""
        tokenizer_input_names = self.tokenizer.model_input_names
        protein_input_names = ["protein_tokenized", "batch_idx_map"]
        return list(dict.fromkeys(tokenizer_input_names + protein_input_names))
    
    def save_pretrained(self, save_directory, **kwargs):
        """保存处理器"""
        self.tokenizer.save_pretrained(save_directory, **kwargs)
        # 可以添加蛋白质tokenizer的保存逻辑
        protein_save_dir = f"{save_directory}/protein_tokenizer"
        self.protein_tokenizer.save_pretrained(protein_save_dir, **kwargs)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """从预训练模型加载处理器"""
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        protein_tokenizer = AutoTokenizer.from_pretrained(
            kwargs.get("protein_model_name", "facebook/esm2_t33_650M_UR50D")
        )
        return cls(tokenizer=tokenizer, protein_tokenizer=protein_tokenizer)


# 使用示例和测试代码
if __name__ == "__main__":
    # 创建处理器
    from transformers import AutoTokenizer
    
    print("Loading tokenizers...")
    text_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B")
    protein_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    
    processor = ProteinLLMProcessor(
        tokenizer=text_tokenizer,
        protein_tokenizer=protein_tokenizer
    )
    
    # 测试数据 (你的实际数据格式)
    test_text = [
        "Does this protein sequence include a signal peptide?\n\nProtein description: A test protein.\nSequence:\n<|protein_pad|>",
        "Analyze this protein: <|protein_pad|> and classify it."
    ]
    
    test_protein_sequences = [
        ["MKLLFLVLMMILSEVYS"],  # 第一个样本的蛋白质序列
        ["MTLSGSGSASDMSGQTV"]   # 第二个样本的蛋白质序列
    ]
    
    print("Processing test data...")
    result = processor(
        batch_protein_sequences=test_protein_sequences,
        text=test_text,
        max_length_text=512,
        max_length_protein=256
    )
    
    print("Processing successful!")
    print(f"Input IDs shape: {result['input_ids'].shape}")
    print(f"Attention mask shape: {result['attention_mask'].shape}")
    print(f"Protein tokenized shape: {result['protein_tokenized']['input_ids'].shape}")
    print(f"Batch idx map: {result['batch_idx_map']}")