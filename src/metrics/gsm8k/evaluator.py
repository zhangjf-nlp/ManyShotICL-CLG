import numpy as np
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)

class EvaluateTool(object):
    def __init__(self):
        pass

    def evaluate(self, preds, golds):
        golds = [gold["answer"] for gold in golds]
        bar = tqdm(zip(preds, golds), total=len(preds), desc="evaluating gsm8k solutions")
        results = []
        for pred, gold in bar:
            gold_ = int(gold.split("\n#### ")[-1].replace(',',''))
            try:
                pred_ = int(pred.strip().split("\n#### ")[-1].replace(',',''))
                result = (gold_==pred_)
            except:
                result = False
            results.append(result)
            bar.set_postfix({
                "pred": pred,
                "gold": gold,
                "result": result
            })
        return {"acc": np.mean(results)}
