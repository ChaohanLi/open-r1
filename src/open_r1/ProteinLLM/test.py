import os
import sys
import torch
import json
from datasets import Dataset
from transformers import TrainingArguments

# 添加项目路径
sys.path.append('/home/x-cli32/chaohan/projects/open-r1/src')

from open_r1.ProteinLLM.ProteinLLMModel import ProteinLLMModel, ProteinLLMConfig
from open_r1.ProteinLLM.ProteinLLMProcessor import ProteinLLMProcessor

# 🔧 修复：使用你的真实数据
def load_real_data():
    """加载你的实际数据"""
    data_path = "/home/x-cli32/chaohan/projects/open-r1/src/open_r1/ProteinLLM/data.jsonl"
    
    samples = []
    with open(data_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    print(f"Loaded {len(samples)} real samples")
    return samples

def test_processor_separately():
    """单独测试Processor功能"""
    print("=== Testing Processor Separately ===")
    
    from transformers import AutoTokenizer
    
    # 加载tokenizers
    text_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B", trust_remote_code=True)
    protein_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
    
    # 添加蛋白质token到文本tokenizer
    text_tokenizer.add_special_tokens({"additional_special_tokens": ["<|protein_pad|>"]})
    
    # 创建processor
    processor = ProteinLLMProcessor(
        tokenizer=text_tokenizer,
        protein_tokenizer=protein_tokenizer
    )
    
    # 测试样本
    test_messages = [
        {
            "role": "user", 
            "content": "Does this protein sequence include a signal peptide?\n\nSequence:\n<|protein_pad|>"
        }
    ]
    
    # 使用chat_template转换
    formatted_text = text_tokenizer.apply_chat_template(
        test_messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    print(f"Formatted text: {formatted_text[:200]}...")
    
    # 测试processor
    result = processor(
        batch_protein_sequences=[["MKIIFLVLMMILSEVYSDRDGYPVHDGTNCKYSCDIREKWEYCTPLCKRRNAKTGYCYAFACWCIGLPDE"]],
        text=[formatted_text],
        max_length_text=512,
        max_length_protein=100
    )
    
    print(f"Processor output keys: {result.keys()}")
    print(f"Input IDs shape: {result['input_ids'].shape}")
    print(f"Protein tokenized shape: {result['protein_tokenized']['input_ids'].shape}")
    print(f"Batch idx map: {result['batch_idx_map']}")
    
    # 🔧 验证ESM token数量
    protein_tokens = result['protein_tokenized']['input_ids'][0]
    attention_mask = result['protein_tokenized']['attention_mask'][0]
    print(f"Protein total tokens: {attention_mask.sum().item()}")
    print(f"Protein tokens (first 10): {protein_tokens[:10].tolist()}")
    
    return processor

def test_model_creation():
    """测试模型创建"""
    print("=== Testing Model Creation ===")
    
    config = ProteinLLMConfig(
        text_model_name="Qwen/Qwen2.5-Math-1.5B",
        protein_model_name="facebook/esm2_t12_35M_UR50D",
        text_model_finetune=True,
        protein_model_finetune=False  # 冻结蛋白质模型
    )
    
    model = ProteinLLMModel(config=config)
    
    print(f"Text model: {model.text_model_name}")
    print(f"Protein model: {model.protein_model_name}")
    print(f"Protein token ID: {model.protein_token_id}")
    print(f"Text vocab size: {len(model.text_tokenizer)}")
    
    return model

def test_forward_pass(model, processor):
    """测试前向传播"""
    print("=== Testing Forward Pass ===")
    
    # 准备测试数据
    test_data = load_real_data()[:2]  # 使用前2个真实样本
    
    # 转换为SFTTrainer期望的格式
    formatted_texts = []
    protein_sequences = []
    
    for sample in test_data:
        # 使用chat_template转换messages
        text = model.text_tokenizer.apply_chat_template(
            sample["messages"], 
            tokenize=False, 
            add_generation_prompt=False
        )
        formatted_texts.append(text)
        protein_sequences.append(sample["protein_sequence"])
    
    print(f"Sample formatted text: {formatted_texts[0][:200]}...")
    
    # 使用processor处理
    batch_inputs = processor(
        batch_protein_sequences=protein_sequences,
        text=formatted_texts,
        max_length_text=512,
        max_length_protein=100,
        return_tensors="pt"
    )
    
    print(f"Batch inputs keys: {batch_inputs.keys()}")
    
    # 测试前向传播
    model.eval()
    with torch.no_grad():
        outputs = model(**batch_inputs)
    
    print(f"Model outputs keys: {outputs.keys()}")
    print(f"Logits shape: {outputs.logits.shape}")
    
    return batch_inputs, outputs

def test_sft_trainer():
    """测试SFTTrainer集成"""
    print("=== Testing SFTTrainer Integration ===")
    
    # 🔧 使用你的真实数据
    real_data = load_real_data()
    dataset = Dataset.from_list(real_data)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample keys: {dataset[0].keys()}")
    
    # 创建模型（使用小模型）
    config = ProteinLLMConfig(
        text_model_name="Qwen/Qwen2.5-Math-1.5B",
        protein_model_name="facebook/esm2_t12_35M_UR50D"
    )
    model = ProteinLLMModel(config=config)
    
    # 训练参数（极简测试）
    training_args = TrainingArguments(
        output_dir="./test_output",
        num_train_epochs=1,
        per_device_train_batch_size=2,  # 小batch size
        learning_rate=1e-5,
        logging_steps=1,
        save_steps=100,
        max_steps=3,  # 只跑3步验证
        dataloader_drop_last=False,
        remove_unused_columns=False,  # 保留protein_sequence列
        gradient_checkpointing=True,  # 节省内存
        fp16=True,  # 使用混合精度
    )
    
    # 🔧 修复：导入正确的SFTTrainer
    try:
        # 尝试从trl导入（如果安装了TRL）
        from trl import SFTTrainer
        print("Using TRL SFTTrainer")
    except ImportError:
        # 如果没有TRL，使用transformers的简单版本
        print("TRL not found, creating simple trainer")
        from transformers import Trainer
        
        class SimpleSFTTrainer(Trainer):
            def __init__(self, processing_class=None, **kwargs):
                super().__init__(**kwargs)
                self.processing_class = processing_class
            
            def _prepare_dataset(self, dataset, tokenizer=None, packing=None, **kwargs):
                def process_sample(examples):
                    # 转换messages为文本
                    texts = []
                    for messages in examples["messages"]:
                        text = self.tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=False
                        )
                        texts.append(text)
                    
                    # 使用processing_class处理
                    processed = self.processing_class(
                        batch_protein_sequences=examples["protein_sequence"],
                        text=texts
                    )
                    
                    # 添加labels（SFT需要）
                    processed["labels"] = processed["input_ids"].clone()
                    
                    return processed
                
                return dataset.map(process_sample, batched=True, remove_columns=dataset.column_names)
        
        SFTTrainer = SimpleSFTTrainer
    
    # 创建训练器
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=model.processor,
        tokenizer=model.text_tokenizer,
    )
    
    print("Starting SFT training test...")
    trainer.train()
    print("SFT training test completed!")

def main():
    """主测试流程"""
    print("🧬 Starting ProteinLLM Open-R1 Integration Test")
    print("=" * 60)
    
    try:
        # Step 1: 测试Processor
        processor = test_processor_separately()
        print("✅ Processor test passed\n")
        
        # Step 2: 测试模型创建  
        model = test_model_creation()
        print("✅ Model creation test passed\n")
        
        # Step 3: 测试前向传播
        batch_inputs, outputs = test_forward_pass(model, processor)
        print("✅ Forward pass test passed\n")
        
        # Step 4: 测试SFTTrainer集成
        test_sft_trainer()
        print("✅ SFTTrainer integration test passed\n")
        
        print("🎉 All tests passed! Ready for full training.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()