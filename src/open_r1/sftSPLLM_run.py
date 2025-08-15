import os
from datasets import load_dataset
from open_r1.sftSPLLM import SPLLMModelArgs
from open_r1.configs import ScriptArguments, SFTConfig
from open_r1.utils import get_spllm_model, get_spllm_data_collator
from trl import SFTTrainer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run():
    # 直接加载数据
    dataset = load_dataset("json", data_files={
        "train": "/home/x-cli32/chaohan/projects/open-r1/src/open_r1/SPLLM/processed_data/train.jsonl",
        "validation": "/home/x-cli32/chaohan/projects/open-r1/src/open_r1/SPLLM/processed_data/validation.jsonl",
    })
    
    training_args = SFTConfig(
        output_dir="/home/x-cli32/chaohan/projects/open-r1/test_sft_output_spllm",
        per_device_train_batch_size=2,
        num_train_epochs=1,
        learning_rate=1e-5,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=1,
        max_seq_length=512,
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
    )
    
    model_args = SPLLMModelArgs(
        model_type="spllm",
        model_name_or_path="Qwen/Qwen3-1.7B",
        protein_model_name="facebook/esm2_t12_35M_UR50D",
        text_model_finetune=True,
        protein_model_finetune=False,
        max_length_protein=70,
        trust_remote_code=True,
    )
    
    # 创建模型
    model = get_spllm_model(
        text_model_name=model_args.model_name_or_path,
        protein_model_name=model_args.protein_model_name,
        text_model_finetune=model_args.text_model_finetune,
        protein_model_finetune=model_args.protein_model_finetune,
    )
    
    # DataCollator
    data_collator = get_spllm_data_collator(
        model=model,
        max_length_text=512,
        max_length_protein=model_args.max_length_protein,
        debug=True,
    )
    
    logger.info("Creating SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        formatting_func=None,  # 关键：让 DataCollator 全权处理原始 messages
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Saving model...")
    trainer.save_model()
    logger.info(f"Model saved to {training_args.output_dir}")

if __name__ == "__main__":
    run()