import json

from src.utils.misc import parallel_run


class EvaluateTool(object):
    def __init__(self):
        pass

    def evaluate(self, preds, golds):
        golds = [gold["logical_form"] for gold in golds]
        #eval_results = parallel_run(eval_single, list(zip(preds, golds)))
        eval_results = [eval_single(args) for args in list(zip(preds, golds))]
        metrics = {"em": sum(eval_results) / len(golds)}
        return metrics


def eval_single(args):
    pred, gold = args
    result = type(pred) is str and pred.strip() == gold
    """
    print(json.dumps(
        {
            "pred": pred,
            "gold": gold,
            "result": result
        },
        ensure_ascii=False,
        indent=4
    ))
    """
    return result
