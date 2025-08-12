"""
测试重新设计的双层架构：Processor + DataCollator
"""
import os
import sys
import torch
import json
import glob
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments
from torch.utils.data import DataLoader
from trl import SFTTrainer

# 添加项目路径
sys.path.append('/home/x-cli32/chaohan/projects/open-r1/src')

import glob
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError

# 添加：更稳健的缓存检查
def _resolve_cached_snapshot(repo_id: str, cache_dirs=None) -> str | None:
    """
    返回本地缓存的 snapshot 绝对路径；若不存在则返回 None。
    优先使用 HF_HOME，其次使用用户 home 下默认路径。
    """
    if cache_dirs is None:
        cache_dirs = [
            os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
            "/home/x-cli32/chaohan/huggingface",  # 你的自定义缓存路径（可放后面作为兜底）
        ]
    for cache_dir in cache_dirs:
        try:
            path = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                local_files_only=True,   # 只查本地，不触网
                resume_download=True,
            )
            if os.path.isdir(path):
                return path
        except LocalEntryNotFoundError:
            continue
        except Exception:
            continue
    return None

def check_model_cache(model_name: str, model_type: str = "any") -> bool:
    """
    更稳健的缓存检查：
      - tokenizer: 接受 tokenizer.json 或 vocab.txt/tokenizer.model 等形式
      - model: 接受 model.safetensors / model-*.safetensors / pytorch_model.bin
      - config: 接受 config.json
      - any: 只要能解析到 snapshot 即认为已缓存
    """
    snapshot_dir = _resolve_cached_snapshot(model_name)
    if snapshot_dir is None:
        return False

    if model_type == "any":
        return True

    # 允许的多种命名与格式
    patterns = {
        "tokenizer": [
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "vocab.json",
            "merges.txt",
            "tokenizer.model",  # sentencepiece
        ],
        "model": [
            "model.safetensors",
            "model-*.safetensors",
            "pytorch_model.bin",
            "tf_model.h5",
            "flax_model.msgpack",
        ],
        "config": ["config.json"],
    }
    required_any = patterns.get(model_type, ["config.json"])
    for pat in required_any:
        if glob.glob(os.path.join(snapshot_dir, pat)):
            return True
    return False

def print_cache_status():
    """打印缓存状态信息（使用新实现）"""
    cache_dir = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    print(f"🔍 HuggingFace缓存目录: {cache_dir}")

    models_to_check = [
        "Qwen/Qwen3-1.7B",
        "facebook/esm2_t12_35M_UR50D",
    ]

    print("\n📦 缓存状态检查:")
    for model in models_to_check:
        any_cached = check_model_cache(model, "any")
        tok_cached = check_model_cache(model, "tokenizer")
        mdl_cached = check_model_cache(model, "model")
        cfg_cached = check_model_cache(model, "config")

        status = "✅ 已缓存" if any_cached else "❌ 未缓存"
        print(f"  {model}: {status}")
        if any_cached:
            print(f"    - Tokenizer files: {'✅' if tok_cached else '❌'}")
            print(f"    - Model weights:   {'✅' if mdl_cached else '❌'}")
            print(f"    - Config:          {'✅' if cfg_cached else '❌'}")
    print()

from open_r1.ProteinLLM.ProteinLLMModel import ProteinLLMModel, ProteinLLMConfig
from open_r1.ProteinLLM.ProteinLLMProcessor import ProteinLLMProcessor
from open_r1.ProteinLLM.ProteinLLMDataCollator import ProteinLLMDataCollator

def load_test_data():
    """加载测试数据"""
    data_path = "/home/x-cli32/chaohan/projects/open-r1/src/open_r1/ProteinLLM/data.jsonl"
    
    samples = []
    with open(data_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    print(f"Loaded {len(samples)} test samples")
    print(f"Sample protein_sequence type: {type(samples[0]['protein_sequence'])}")
    print(f"Sample protein_sequence length: {len(samples[0]['protein_sequence'])}")
    return samples  

def test_processor_alone():
    """测试Processor单独工作"""
    print("=== 测试Processor单独工作 ===")
    
    # 检查缓存状态
    text_model = "Qwen/Qwen3-1.7B"
    protein_model = "facebook/esm2_t12_35M_UR50D"
    
    text_cached = check_model_cache(text_model, "tokenizer")
    protein_cached = check_model_cache(protein_model, "tokenizer")
    
    print(f"缓存检查:")
    print(f"  {text_model}: {'✅ 已缓存' if text_cached else '❌ 未缓存，需要下载'}")
    print(f"  {protein_model}: {'✅ 已缓存' if protein_cached else '❌ 未缓存，需要下载'}")
    
    if not text_cached or not protein_cached:
        print("⚠️ 部分模型未缓存，可能遇到429错误")
    
    try:
        # 1. 准备tokenizers
        print("加载tokenizers...")
        text_tokenizer = AutoTokenizer.from_pretrained(
            text_model, 
            trust_remote_code=True,
            local_files_only=False  # 允许在线下载，但优先使用缓存
        )
        protein_tokenizer = AutoTokenizer.from_pretrained(
            protein_model,
            local_files_only=False  # 允许在线下载，但优先使用缓存
        )
        
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
        print(f"Sample text length: {len(texts[0])}")
        print(f"Sample text length: {len(texts[1])}")
        print(f"Sample text length: {len(texts[2])}")
        print(f"Sample protein: {protein_sequences[0][:20]}...")
        
        # 5. 测试Processor
        result = processor(
            text=texts,
            protein_sequence=protein_sequences,
            max_length_text=512,
            max_length_protein=72,
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
    
    # 检查模型缓存状态
    text_model = "Qwen/Qwen3-1.7B"
    protein_model = "facebook/esm2_t12_35M_UR50D"
    
    text_model_cached = check_model_cache(text_model, "model")
    protein_model_cached = check_model_cache(protein_model, "model")
    
    print(f"模型缓存检查:")
    print(f"  {text_model}: {'✅ 已缓存' if text_model_cached else '❌ 未缓存，需要下载'}")
    print(f"  {protein_model}: {'✅ 已缓存' if protein_model_cached else '❌ 未缓存，需要下载'}")
    
    if not text_model_cached or not protein_model_cached:
        print("⚠️ 部分模型未缓存，可能遇到429错误，跳过此测试")
        print("💡 可以先运行SFTTrainer测试，它会下载并缓存模型")
        return None, None
    
    # 1. 创建模型（为了获取processor）- 使用与SFTTrainer相同的配置
    config = ProteinLLMConfig(
        text_model_name=text_model,
        protein_model_name=protein_model,
        text_model_finetune=True,
        protein_model_finetune=False
    )
    
    try:
        print("创建模型（从缓存加载）...")
        model = ProteinLLMModel(config=config)
        print(f"✅ 模型创建成功")
        
        # 2. 创建DataCollator
        data_collator = ProteinLLMDataCollator(
            processor=model.processor,
            max_length_text=512,
            max_length_protein=72
        )
        
        # 3. 准备数据
        test_data = load_test_data()
        dataset = Dataset.from_list(test_data)
        
        # 4. 测试DataCollator
        batch_features = [dataset[i] for i in range(5)]  # 取前5个样本
        
        batch = data_collator(batch_features)
        
        print(f"✅ DataCollator成功！")
        print(f"Batch keys: {batch.keys()}")
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        if batch.get('protein_tokenized'):
            print(f"Protein shape: {batch['protein_tokenized']['input_ids'].shape}")
        print(f"Labels shape: {batch['labels'].shape}")
        
        # 5. 用 PyTorch DataLoader 测试“两个 batch”：batch_size=5
        print("\n=== 用 DataLoader 测试批次（batch_size=5，预期2个batch）===")
        dl = DataLoader(dataset, batch_size=5, shuffle=False, collate_fn=data_collator)
        num_batches = 0
        for step, b in enumerate(dl):
            num_batches += 1
            print(f"[DataLoader] step={step} -> input_ids {tuple(b['input_ids'].shape)}")
        print(f"DataLoader 总批次数: {num_batches}（应为2）")

        return data_collator, batch
        
    except Exception as e:
        print(f"❌ DataCollator测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def debug_embedding_fusion():
    """专门调试embedding融合问题"""
    print("🔍 调试Embedding融合...")
    
    config = ProteinLLMConfig(
        text_model_name="Qwen/Qwen3-1.7B",
        protein_model_name="facebook/esm2_t12_35M_UR50D"
    )
    model = ProteinLLMModel(config=config)
    test_data = load_test_data()
    
    data_collator = ProteinLLMDataCollator(
        processor=model.processor,
        max_length_text=512,
        max_length_protein=72
    )
    
    batch = data_collator([test_data[0]])
    
    # 🔧 逐步调试
    print("1. 检查蛋白质tokenization...")
    protein_tokenized = batch['protein_tokenized']
    print(f"蛋白质tokens: {protein_tokenized['input_ids'].shape}")
    print(f"蛋白质tokens范围: {protein_tokenized['input_ids'].min()} - {protein_tokenized['input_ids'].max()}")
    
    print("2. 检查ESM编码...")
    with torch.no_grad():
        protein_embeddings = model.encode_protein_sequences(protein_tokenized)
        if protein_embeddings is not None:
            print(f"蛋白质embeddings: {protein_embeddings.shape}")
            print(f"蛋白质embeddings统计: mean={protein_embeddings.mean().item():.4f}, std={protein_embeddings.std().item():.4f}")
            print(f"蛋白质embeddings是否有NaN: {torch.isnan(protein_embeddings).any()}")
        else:
            print("🚨 蛋白质embeddings为None!")
    
    print("3. 检查文本embeddings...")
    text_embeddings = model.text_model.get_input_embeddings()(batch['input_ids'])
    print(f"文本embeddings: {text_embeddings.shape}")
    print(f"文本embeddings统计: mean={text_embeddings.mean().item():.4f}, std={text_embeddings.std().item():.4f}")
    print(f"文本embeddings是否有NaN: {torch.isnan(text_embeddings).any()}")
    
    print("4. 🔍 关键步骤：检查fusion过程...")
    print(f"🔧 检查protein_token_id: {model.protein_token_id}")
    protein_positions = (batch['input_ids'] == model.protein_token_id)
    print(f"🔧 发现{protein_positions.sum().item()}个蛋白质占位符位置")
    
    if protein_positions.sum().item() > 0:
        print("🔧 检查fusion前的状态...")
        print(f"batch['input_ids'].shape: {batch['input_ids'].shape}")
        print(f"batch['attention_mask'].shape: {batch['attention_mask'].shape}")
        print(f"protein_embeddings.shape: {protein_embeddings.shape}")
        fused_embeddings, _ = model.fuse_protein_embeddings(
            batch['input_ids'], 
            batch['attention_mask'], 
            protein_embeddings, 
            batch['batch_idx_map']
        )
        print(f"🔧 Fusion后embeddings形状: {fused_embeddings.shape}")
        print(f"🔧 Fusion后统计: mean={fused_embeddings.mean().item():.4f}, std={fused_embeddings.std().item():.4f}")
        print(f"🔧 Fusion后是否有NaN: {torch.isnan(fused_embeddings).any()}")
        
        if torch.isnan(fused_embeddings).any():
            print("🚨 NaN在fusion步骤中产生！")
            # 检查具体哪些位置有NaN
            nan_positions = torch.isnan(fused_embeddings).any(dim=-1)
            print(f"🔧 NaN位置数量: {nan_positions.sum().item()}")
            print(f"🔧 NaN位置索引: {nan_positions.nonzero().flatten().tolist()}")
        else:
            print("✅ Fusion步骤正常")
            
            print("5. 🔍 检查模型forward中间步骤...")
            # 手动执行forward的各个步骤
            model.eval()
            
            # Step 1: 融合后直接传入text_model
            print("🔧 Step 1: 将融合embeddings传入text_model...")
            try:
                text_outputs = model.text_model(
                    inputs_embeds=fused_embeddings,
                    attention_mask=batch['attention_mask'],
                    return_dict=True
                )
                
                print(f"🔧 Text model输出logits形状: {text_outputs.logits.shape}")
                print(f"🔧 Text model logits统计: mean={text_outputs.logits.mean().item():.4f}, std={text_outputs.logits.std().item():.4f}")
                print(f"🔧 Text model logits是否有NaN: {torch.isnan(text_outputs.logits).any()}")
                
                if torch.isnan(text_outputs.logits).any():
                    print("🚨 NaN在text_model forward中产生！")
                else:
                    print("✅ Text model forward正常")
                
            except Exception as e:
                print(f"🚨 Text model forward失败: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("⚠️ 没有找到蛋白质占位符，可能tokenization有问题")

def test_sft_trainer_integration():
    """测试SFTTrainer集成 - 真实版本"""
    print("\n=== 测试SFTTrainer集成 - 真实版本 ===")
    # ...existing import...
    
    # 🔧 使用 10 条样本用于两个 batch
    test_data = load_test_data()
    print(f"✅ 测试数据加载完成，共{len(test_data)}个样本")
    
    # 1. 创建模型
    print("创建双模态模型...")
    config = ProteinLLMConfig(
        text_model_name="Qwen/Qwen3-1.7B",
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
            max_length_protein=72
        )
        print(f"✅ DataCollator创建成功")
        
        # 3. 准备数据集（10条）
        dataset = Dataset.from_list(test_data)
        print(f"✅ 数据集准备完成，共{len(dataset)}个样本")

        # 4. 配置训练参数：batch=5，1个epoch（应只有2个step）
        training_args = TrainingArguments(
            output_dir="./test_sft_output",
            per_device_train_batch_size=5,      # 关键：每批5条
            gradient_accumulation_steps=1,
            num_train_epochs=1,                 # 用epoch而不是max_steps
            learning_rate=1e-5,
            logging_steps=1,
            save_strategy="no",
            eval_strategy="no",
            gradient_checkpointing=True,
            fp16=False,
            bf16=True,
            remove_unused_columns=False,
            dataloader_drop_last=False,         # 不丢弃，10样本 → 2个batch
            report_to=["wandb"],
            dataloader_num_workers=0,
            ddp_find_unused_parameters=False,
        )
        print(f"✅ 训练参数配置完成（batch=5, epochs=1）")
        
        # 重要：避免 SFTTrainer 预处理覆盖你的 DataCollator
        model.name_or_path = config.text_model_name
        model.config.name_or_path = config.text_model_name

        # 5. 创建SFTTrainer
        print("创建SFTTrainer...")
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            formatting_func=None,   # 保持None，DataCollator全权处理
            # 不传 processing_class
        )
        print(f"✅ SFTTrainer创建成功")
        
        # 6. 检查 dataloader 批次数（应为2）
        print("\n🔍 调试Trainer数据加载...")
        train_dataloader = trainer.get_train_dataloader()
        print(f"Debug dataloader length: {len(train_dataloader)}（应为2）")
        first_batch = next(iter(train_dataloader))
        print(f"首个batch input_ids shape: {tuple(first_batch['input_ids'].shape)}（应为(5, T)）")
        print("🔍 Trainer数据加载调试结束\n")
        
        # 7. 可选：跑一个非常短的训练以验证管道（会有2个step）
        print("\n🚀 开始SFT训练测试（期望2个step）...")
        train_result = trainer.train()
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
    
    # 首先打印缓存状态
    print_cache_status()
    
    # 测试Processor
    processor, processor_result = test_processor_alone()
    if processor_result is not None:
        print("✅ Processor测试通过\\n")
    
    # 测试DataCollator
    data_collator, batch = test_datacollator_with_processor()
    if batch is not None:
        print("✅ DataCollator测试通过\\n")
    
    debug_embedding_fusion()

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
    else:
        print("\\n💡 建议:")
        if processor_result is None:
            print("- Processor失败：检查tokenizer是否正确加载")
        if batch is None:
            print("- DataCollator跳过：等待模型缓存完成后重试")
        if train_result is None:
            print("- SFTTrainer失败：检查模型和训练配置")

if __name__ == "__main__":
    main()
