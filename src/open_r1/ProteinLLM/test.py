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
    """测试SFTTrainer集成 - 真实版本"""
    print("\\n=== 测试SFTTrainer集成 - 真实版本 ===")
    
    try:
        from trl import SFTTrainer
        print("✅ 成功导入SFTTrainer")
    except ImportError:
        print("❌ 无法导入SFTTrainer，请安装TRL库")
        return None, None
    
    # 1. 创建模型
    print("创建双模态模型...")
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
        print(f"✅ DataCollator创建成功")
        
        # 3. 准备数据集
        test_data = load_test_data()
        dataset = Dataset.from_list(test_data)
        print(f"✅ 数据集准备完成，共{len(dataset)}个样本")
        
        # 4. 配置训练参数
        training_args = TrainingArguments(
            output_dir="./test_sft_output",
            max_steps=5,  # 只训练5步来测试
            per_device_train_batch_size=1,  # 小batch size避免内存问题
            learning_rate=1e-5,
            logging_steps=1,
            save_steps=10,
            gradient_checkpointing=False,
            fp16=False,  # 避免数值问题
            bf16=False,
            remove_unused_columns=False,  # 保留protein_sequence列
            dataloader_drop_last=False,
            eval_strategy="no",  # 不进行评估
            save_strategy="no",   # 不保存checkpoint
        )
        print(f"✅ 训练参数配置完成")
        
        # 5. 创建SFTTrainer - 正确的双模态配置
        print("创建SFTTrainer...")
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,        # 🔧 关键：让DataCollator处理所有数据
            # 🔧 重要：完全禁用SFTTrainer的数据预处理
            formatting_func=None,              # 不使用formatting函数 - 让DataCollator全权处理
            # 不传入processing_class，避免SFTTrainer调用Processor
        )
        print(f"✅ SFTTrainer创建成功")
        
        # 6. 运行训练测试
        print("开始训练测试（5步）...")
        train_result = trainer.train()
        
        print(f"✅ SFTTrainer训练测试成功！")
        print(f"训练指标: {train_result.metrics}")
        
        return trainer, train_result
        
    except Exception as e:
        print(f"❌ SFTTrainer集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

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
    
    # 测试SFTTrainer集成 - 真实版本
    trainer, train_result = test_sft_trainer_integration()
    if train_result is not None:
        print("✅ SFTTrainer集成测试通过\\n")
    
    print("\\n🎉 完整架构测试完成！")
    
    # 总结报告
    print("\\n📊 测试总结:")
    print(f"- Processor: {'✅ 通过' if processor_result is not None else '❌ 失败'}")
    print(f"- DataCollator: {'✅ 通过' if batch is not None else '❌ 失败'}")
    print(f"- SFTTrainer: {'✅ 通过' if train_result is not None else '❌ 失败'}")
    
    if all([processor_result is not None, batch is not None, train_result is not None]):
        print("\\n🏆 所有测试通过！双模态架构准备就绪！")

if __name__ == "__main__":
    main()
