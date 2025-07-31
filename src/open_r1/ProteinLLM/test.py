"""
测试重新设计的双层架构：Processor + DataCollator
"""
import os
import sys
import torch
import json
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments

# 添加项目路径
sys.path.append('/home/cl426/data/open-r1/src')

from open_r1.ProteinLLM.ProteinLLMModel import ProteinLLMModel, ProteinLLMConfig
from open_r1.ProteinLLM.ProteinLLMProcessor import ProteinLLMProcessor
from open_r1.ProteinLLM.ProteinLLMDataCollator import ProteinLLMDataCollator

def load_test_data():
    """加载测试数据"""
    data_path = "/home/cl426/data/open-r1/src/open_r1/ProteinLLM/data.jsonl"
    
    samples = []
    with open(data_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    print(f"Loaded {len(samples)} test samples")
    print(f"Sample protein_sequence type: {type(samples[0]['protein_sequence'])}")
    print(f"Sample protein_sequence length: {len(samples[0]['protein_sequence'])}")
    return samples[:3]  # 只用前3个样本测试

def test_processor_alone():
    """测试Processor单独工作"""
    print("=== 测试Processor单独工作 ===")
    
    # 1. 准备tokenizers
    text_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B", trust_remote_code=True)
    protein_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
    
    # 2. 添加特殊token
    text_tokenizer.add_special_tokens({"additional_special_tokens": ["<|protein_pad|>"]})
    
    # 3. 创建Processor
    processor = ProteinLLMProcessor(
        tokenizer=text_tokenizer,
        protein_tokenizer=protein_tokenizer
    )
    
    # 4. 准备测试数据
    test_data = load_test_data()
    
    # 提取文本和蛋白质序列
    texts = []
    protein_sequences = []
    
    for sample in test_data:
        # 使用chat_template处理messages
        text = text_tokenizer.apply_chat_template(
            sample["messages"], 
            tokenize=False, 
            add_generation_prompt=False
        )
        texts.append(text)
        protein_sequences.append(sample["protein_sequence"])  # 现在是字符串
    
    print(f"Sample text: {texts[0][:100]}...")
    print(f"Sample protein: {protein_sequences[0][:20]}...")
    
    # 5. 测试Processor
    try:
        result = processor(
            text=texts,
            protein_sequence=protein_sequences,
            max_length_text=512,
            max_length_protein=100,
            return_tensors="pt"
        )
        
        print(f"✅ Processor成功！")
        print(f"Result keys: {result.keys()}")
        print(f"Text shape: {result['input_ids'].shape}")
        print(f"Protein shape: {result['protein_tokenized']['input_ids'].shape}")
        print(f"Batch idx map: {result['batch_idx_map']}")
        
        return processor, result
        
    except Exception as e:
        print(f"❌ Processor失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_datacollator_with_processor():
    """测试DataCollator与Processor的协作"""
    print("\\n=== 测试DataCollator与Processor协作 ===")
    
    # 1. 创建模型（为了获取processor）
    config = ProteinLLMConfig(
        text_model_name="Qwen/Qwen2.5-Math-1.5B",
        protein_model_name="facebook/esm2_t12_35M_UR50D",
        text_model_finetune=True,
        protein_model_finetune=False
    )
    
    try:
        model = ProteinLLMModel(config=config)
        print(f"✅ 模型创建成功")
        
        # 2. 创建DataCollator
        data_collator = ProteinLLMDataCollator(
            processor=model.processor,
            max_length_text=512,
            max_length_protein=100
        )
        
        # 3. 准备数据
        test_data = load_test_data()
        dataset = Dataset.from_list(test_data)
        
        # 4. 测试DataCollator
        batch_features = [dataset[i] for i in range(min(2, len(dataset)))]
        
        batch = data_collator(batch_features)
        
        print(f"✅ DataCollator成功！")
        print(f"Batch keys: {batch.keys()}")
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        if batch.get('protein_tokenized'):
            print(f"Protein shape: {batch['protein_tokenized']['input_ids'].shape}")
        print(f"Labels shape: {batch['labels'].shape}")
        
        return data_collator, batch
        
    except Exception as e:
        print(f"❌ DataCollator测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_sft_trainer_integration():
    """测试SFTTrainer集成"""
    print("\\n=== 测试SFTTrainer集成 ===")
    
    # 这里可以测试真实的SFTTrainer集成
    # 但为了避免下载大模型，我们先跳过
    print("SFTTrainer集成测试暂时跳过（避免下载大模型）")
    
    # 推荐的配置：
    print("\\n推荐的SFTTrainer配置：")
    print("""
    trainer = SFTTrainer(
        model=protein_llm_model,           # 自定义模型
        args=training_args,
        train_dataset=dataset,
        processing_class=protein_processor,  # 🔧 传入Processor
        data_collator=protein_data_collator, # 🔧 传入DataCollator
    )
    """)

def main():
    """主测试流程"""
    print("🧬 测试重新设计的双层架构")
    print("=" * 60)
    
    # 测试Processor
    processor, processor_result = test_processor_alone()
    if processor_result is not None:
        print("✅ Processor测试通过\\n")
    
    # 测试DataCollator
    data_collator, batch = test_datacollator_with_processor()
    if batch is not None:
        print("✅ DataCollator测试通过\\n")
    
    # 测试SFTTrainer集成
    test_sft_trainer_integration()
    
    print("\\n🎉 架构测试完成！")

if __name__ == "__main__":
    main()
