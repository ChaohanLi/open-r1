"""
蛋白质双模态模型训练脚本
基于 sft.py 扩展
"""
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import ScriptArguments, SFTConfig
from open_r1.utils import get_protein_llm_model, get_protein_data_collator
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import ModelConfig, SFTTrainer, TrlParser, get_peft_config

logger = logging.getLogger(__name__)

@dataclass
class ProteinModelConfig(ModelConfig):
    """蛋白质双模态模型配置"""
    model_type: str = field(default="protein_llm", metadata={"help": "Model type"})
    protein_model_name: str = field(default="facebook/esm2_t12_35M_UR50D", metadata={"help": "Protein model name"})
    text_model_finetune: bool = field(default=True, metadata={"help": "Whether to finetune text model"})
    protein_model_finetune: bool = field(default=False, metadata={"help": "Whether to finetune protein model"})
    max_length_protein: int = field(default=100, metadata={"help": "Max protein sequence length"})

def main(script_args, training_args, model_args):
    set_seed(training_args.seed)
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    logger.info(f"Protein Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Load dataset and model
    dataset = get_dataset(script_args)
    
    # 🔧 创建双模态模型
    model = get_protein_llm_model(
        text_model_name=model_args.model_name_or_path,
        protein_model_name=model_args.protein_model_name,
        text_model_finetune=model_args.text_model_finetune,
        protein_model_finetune=model_args.protein_model_finetune,
    )
    
    # 🔧 创建数据收集器
    data_collator = get_protein_data_collator(
        model=model,
        max_length_text=training_args.max_seq_length or 512,
        max_length_protein=model_args.max_length_protein,
    )

    # Initialize the SFT Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None),
        data_collator=data_collator,  # 🔧 使用自定义数据收集器
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        formatting_func=None,  # 🔧 禁用默认格式化，让DataCollator处理
    )

    # Training loop
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    # Save and evaluate
    trainer.save_model(training_args.output_dir)

    try:
        model.text_tokenizer.save_pretrained(os.path.join(training_args.output_dir, "text_tokenizer"))
        model.processor.save_pretrained(os.path.join(training_args.output_dir, "processor"))
    except Exception as e:
        logger.warning(f"Saving processor/tokenizer failed: {e}")
        
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ProteinModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)