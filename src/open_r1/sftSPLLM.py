"""
SFT 冷启动脚本（SPLLM：仅 messages，内嵌蛋白序列）
"""
import logging, os, sys
import datasets, transformers
from dataclasses import dataclass, field
from typing import Optional

from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer, TrlParser, ModelConfig, get_peft_config, setup_chat_format

from open_r1.configs import ScriptArguments, SFTConfig
from open_r1.utils import get_dataset
from open_r1.utils import get_spllm_model, get_spllm_data_collator

logger = logging.getLogger(__name__)

@dataclass
class SPLLMModelArgs(ModelConfig):
    model_type: str = field(default="spllm", metadata={"help": "Model type"})
    protein_model_name: str = field(default="facebook/esm2_t12_35M_UR50D")
    text_model_finetune: bool = field(default=True)
    protein_model_finetune: bool = field(default=False)
    max_length_protein: int = field(default=72)

def _count_params(model):
    t = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return t, tr

def main(script_args, training_args, model_args):
    set_seed(training_args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # 数据
    dataset = get_dataset(script_args)

    # 模型
    model = get_spllm_model(
        text_model_name=model_args.model_name_or_path,
        protein_model_name=model_args.protein_model_name,
        text_model_finetune=model_args.text_model_finetune,
        protein_model_finetune=model_args.protein_model_finetune,
    )

    total, trainable = _count_params(model)
    logger.info(f"Params total={total/1e6:.1f}M, trainable={trainable/1e6:.1f}M "
            f"(text_model_finetune={model.config.text_model_finetune}, ESM frozen, projection trainable)")

    # chat 模板（仅用于渲染 messages → text）
    if model.text_tokenizer.chat_template is None:
        logger.info("No chat template provided, defaulting to ChatML.")
        model.text_model, model.text_tokenizer = setup_chat_format(model.text_model, model.text_tokenizer, format="chatml")

    # DataCollator
    data_collator = get_spllm_data_collator(
        model=model,
        max_length_text=training_args.max_seq_length or 512,
        max_length_protein=model_args.max_length_protein,
    )

    # Trainer（关键：不传 processing_class；formatting_func=None；remove_unused_columns=False）
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None),
        data_collator=data_collator,
        formatting_func=None,
        peft_config=get_peft_config(model_args),
    )

    # 训练
    logger.info("*** Train ***")
    checkpoint = training_args.resume_from_checkpoint or last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # 保存
    logger.info("*** Save model ***")
    trainer.model.generation_config.eos_token_id = model.text_tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)
    try:
        model.text_tokenizer.save_pretrained(os.path.join(training_args.output_dir, "text_tokenizer"))
        model.processor.save_pretrained(os.path.join(training_args.output_dir, "processor"))
    except Exception as e:
        logger.warning(f"save processor/tokenizer failed: {e}")
    logger.info(f"Saved to {training_args.output_dir}")

    # 评估（可选）
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, SPLLMModelArgs))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
