from typing import Optional

from transformers import ViltProcessor, AutoTokenizer


class VaultProcessor(ViltProcessor):
    @classmethod
    def from_pretrained(
        self, vilt_directory: str, bert_directory: Optional[str] = None
    ):
        try:
            processor = super().from_pretrained(vilt_directory)
        except:
            path_vilt_b32_mlm = '/home/mayrain/model/vilt-b32-mlm'
            # not all processors have been implemented
            processor = super().from_pretrained(path_vilt_b32_mlm)
        if bert_directory is not None:
            processor.tokenizer = AutoTokenizer.from_pretrained(bert_directory)
        return processor
