import evaluate
import logging
logger = logging.getLogger(__name__)


def find_last_abcd(s):
    for char in reversed(s):
        if char in {'A', 'B', 'C', 'D'}:
            return char
    return 'Z'


class EvaluateTool(object):
    def __init__(self):
        pass

    def evaluate(self, preds, golds):
        golds = [ord(gold["Answer"])-ord('A') for gold in golds]
        preds = [ord(find_last_abcd(pred))-ord('A') for pred in preds]
        metric = evaluate.load("evaluate/metrics/accuracy/accuracy.py")
        return metric.compute(references=golds, predictions=preds)
