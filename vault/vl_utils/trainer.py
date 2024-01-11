from typing import Iterable, Any, Dict, List

import torch
import numpy as np

from vault.tmsc_utils.trainer import Twitter201XTrainer


class VisionAndLanguageTrainer(Twitter201XTrainer):

    def input_batch_kwargs(self, batch: Iterable[Any]) -> Dict[str, Any]:
        return batch[0]

    def get_logits_from_model(self, return_vals: Any, *args, **kwargs):
        return return_vals.logits.squeeze()

    def batch_len(self, batch):
        return len(next(iter(batch[0].values())))

    def get_eval_preds_from_batch(self, logits: torch.Tensor) -> List[int]:
        return [ex_logits.argmax(dim=-1).item() for ex_logits in logits]

    def get_eval_true_from_batch(self, labels: torch.Tensor) -> List[int]:
        return labels.tolist()

    def evaluation_metrics(
        self, eval_true: List[int], eval_preds: List[int], data_loader=None
    ) -> Dict[str, float]:
        eval_accuracy = np.mean(
            [pred == label for pred, label in zip(eval_preds, eval_true)]
        )

        return dict(eval_accuracy=eval_accuracy)
