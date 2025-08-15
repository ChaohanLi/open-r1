import re
from typing import List, Dict, Any
import torch

INLINE_PATTERNS = [
    re.compile(r"<\|protein_start\|>\s*([A-Za-z]+)\s*<\|protein_end\|>"),
    re.compile(r"<protein>\s*([A-Za-z]+)\s*</protein>"),
]

def _find_last_subseq(hay: List[int], needle: List[int]) -> int:
    if not needle:
        return -1
    n, m = len(hay), len(needle)
    for i in range(n - m, -1, -1):
        if hay[i:i+m] == needle:
            return i
    return -1

class SPLLMDataCollator:
    """
    - 渲染 messages -> 文本
    - 抽取内嵌蛋白序列；文本中以一个占位符 token <|protein_pad|> 标记位置
    - 调用 Processor 做双模态编码与 token 级占位扩展（固定 70 aa）
    - 仅对 assistant 段计算 loss（系统/用户/占位符/模板均屏蔽）
    """
    def __init__(self, processor, max_length_text: int = 512, max_length_protein: int = 70, debug: bool = False):
        self.processor = processor
        self.max_length_text = max_length_text
        self.max_length_protein = max_length_protein
        self.debug = debug

        tok = self.processor.tokenizer
        self.protein_token = getattr(processor, "protein_token", "<|protein_pad|>")
        self.protein_token_id = tok.convert_tokens_to_ids(self.protein_token)
        self.assist_hdr_ids = tok.encode("<|im_start|>assistant\n", add_special_tokens=False) or \
                              tok.encode("<|im_start|>assistant", add_special_tokens=False)
        self.im_end_ids = tok.encode("<|im_end|>", add_special_tokens=False)
        self.pad_id = tok.pad_token_id

    def _render_messages(self, messages: List[Dict[str, str]]) -> str:
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    def _extract_inline(self, text: str) -> (str, str):
        for pat in INLINE_PATTERNS:
            m = pat.search(text)
            if m:
                seq = m.group(1).strip().upper()
                new_text = text[:m.start()] + self.protein_token + text[m.end():]
                return new_text, seq
        raise ValueError("Inline protein sequence not found. Expect <|protein_start|>SEQ<|protein_end|> or <protein>SEQ</protein>.")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts, seqs = [], []
        for i, f in enumerate(features):
            raw = self._render_messages(f["messages"]) if "messages" in f else f["text"]
            t, s = self._extract_inline(raw)
            texts.append(t)
            seqs.append(s)

        # Processor: 挖坑扩展 + 双分词（固定 70 aa）
        batch = self.processor(
            text=texts,
            protein_sequence=seqs,
            max_length_text=self.max_length_text,
            max_length_protein=self.max_length_protein,
            return_tensors="pt",
        )
        batch = dict(batch)

        # 仅对 assistant 段计算 loss
        input_ids = batch["input_ids"]
        labels = input_ids.clone()

        for i in range(input_ids.size(0)):
            ids_i = input_ids[i].tolist()
            start = _find_last_subseq(ids_i, self.assist_hdr_ids)
            if start == -1:
                labels[i] = torch.full_like(labels[i], -100)
                continue
            start += len(self.assist_hdr_ids)
            end = _find_last_subseq(ids_i, self.im_end_ids)
            if end == -1 or end < start:
                end = len(ids_i)

            labels[i] = torch.full_like(labels[i], -100)
            keep_len = max(0, min(end, input_ids.size(1)) - start)
            if keep_len > 0:
                labels[i, start:start+keep_len] = input_ids[i, start:start+keep_len]

        if self.pad_id is not None:
            labels[input_ids == self.pad_id] = -100

        batch["labels"] = labels

        if self.debug:
            print(f"[SPLLMDataCollator] input_ids={tuple(batch['input_ids'].shape)} "
                  f"protein={tuple(batch['protein_tokenized']['input_ids'].shape)}")
        return batch