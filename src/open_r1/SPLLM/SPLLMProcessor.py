from typing import List, Optional, Union
from transformers import ProcessorMixin, BatchFeature

class SPLLMProcessor(ProcessorMixin):
    """
    - 两个 tokenizer（文本 + 蛋白）
    - 先蛋白侧 tokenize（截断=70），得到 aa_count
    - 文本侧先含一个占位符 token；在 token 级将该占位符扩展为 aa_count 个
    """
    attributes = ["tokenizer", "protein_tokenizer"]
    valid_kwargs = ["model"]
    tokenizer_class = ("PreTrainedTokenizer", "PreTrainedTokenizerFast")
    protein_tokenizer_class = ("PreTrainedTokenizer", "PreTrainedTokenizerFast")

    def __init__(self, tokenizer=None, protein_tokenizer=None, protein_token: str = "<|protein_pad|>", **kwargs):
        self.tokenizer = tokenizer
        self.protein_tokenizer = protein_tokenizer
        self.protein_token = protein_token
        super().__init__(tokenizer, protein_tokenizer)
        if not getattr(self.tokenizer, "pad_token", None):
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def apply_chat_template(self, *a, **k):  # 供 collator 调用
        return self.tokenizer.apply_chat_template(*a, **k)

    def __call__(
        self,
        text: Union[str, List[str]],
        protein_sequence: Union[str, List[str]],
        max_length_text: int = 512,
        max_length_protein: int = 70,
        return_tensors: Optional[str] = "pt",
        **kwargs,
    ) -> BatchFeature:
        if isinstance(text, str):
            text = [text]
        if isinstance(protein_sequence, str):
            protein_sequence = [protein_sequence]
        if len(text) != len(protein_sequence):
            raise ValueError(f"text and protein_sequence length mismatch: {len(text)} vs {len(protein_sequence)}")

        # 1) 蛋白侧 tokenize（固定 70）
        prot = self.protein_tokenizer(
            protein_sequence,
            padding=True,
            truncation=True,
            max_length=max_length_protein,
            return_tensors=return_tensors,
            return_attention_mask=True,
        )
        # 去掉 [CLS]/[EOS]
        aa_counts: List[int] = []
        for i in range(prot["attention_mask"].size(0)):
            total = int(prot["attention_mask"][i].sum().item())
            aa_counts.append(max(1, total - 2))

        # 2) 文本侧 tokenize（先仅 1 个占位符 token）
        enc = self.tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=max_length_text,
            return_attention_mask=True,
        )
        prot_token_id = self.tokenizer.convert_tokens_to_ids(self.protein_token)

        def expand_ids(ids: List[int], mask: List[int], n: int):
            out_ids, out_mask = [], []
            for tid, m in zip(ids, mask):
                if tid == prot_token_id:
                    out_ids.extend([prot_token_id] * n)
                    out_mask.extend([m] * n)
                else:
                    out_ids.append(tid)
                    out_mask.append(m)
            return out_ids, out_mask

        samples = []
        for ids, am, n in zip(enc["input_ids"], enc["attention_mask"], aa_counts):
            new_ids, new_am = expand_ids(ids, am, n)
            samples.append({"input_ids": new_ids, "attention_mask": new_am})

        text_tok = self.tokenizer.pad(samples, padding=True, max_length=None, return_tensors=return_tensors)

        return BatchFeature(
            data={
                **text_tok,
                "protein_tokenized": prot,
                "batch_idx_map": list(range(len(text))),
            }
        )