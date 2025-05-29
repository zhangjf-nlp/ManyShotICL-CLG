# encoding=utf8
from datasets import DownloadConfig

import evaluate


class EvaluateTool:
    def __init__(self):
        self.bleu = evaluate.load(
            path="evaluate/metrics/bleu",
            download_config=DownloadConfig(
                cache_dir='evaluate/cache/bleu',
                local_files_only=True,
            )
        )

    def evaluate(self, preds, golds):
        # character-level 4-bleu
        predictions = [" ".join(ch for ch in text) for text in preds]
        references = [[" ".join(ch for ch in entry['bash'])] for entry in golds]
        return self.bleu.compute(predictions=predictions, references=references)