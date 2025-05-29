import hashlib
import json
import time
import torch
from transformers import TrainerCallback, AutoTokenizer, AutoConfig
from modeling_llama_prefix_tuning import LlamaForPrefixTuning
from modeling_qwen2_prefix_tuning import Qwen2ForPrefixTuning
from src.dataset_readers.dataset_wrappers import get_dataset_wrapper
from utils_training import get_basic_parser, get_trainer, check_output_dir, set_seed, get_last_checkpoint


def batch_padding(list_ids, padding_side="left", pad_token_id=-100):
    lengths = [len(ids) for ids in list_ids]
    max_length = max(lengths)
    padding_length = [max_length - length for length in lengths]
    def pad(ids, pl, side, pt):
        if side == "left":
            return [pt] * pl + ids
        elif side == "right":
            return ids + [pt] * pl
        else:
            raise NotImplementedError(side)
    padded_ids = [
        pad(ids, pl, padding_side, pad_token_id)
        for ids, pl in zip(list_ids, padding_length)
    ]
    return padded_ids


def model_name_to_max_length(model_name):
    if "llama" in model_name.lower():
        return 512
    elif "qwen2" in model_name.lower():
        return 1024
    else:
        raise NotImplementedError(model_name)


class DatasetForCVAE:
    def __init__(self, task_name, usage, model_name):
        self.task_name = task_name
        self.usage = usage
        self.model_name = model_name
        self.max_length = model_name_to_max_length(model_name)
        self.dsw = get_dataset_wrapper(
            task_name,
            dataset_split=usage,
            ds_size=(50000 if usage == "train" else 10000)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.prepare()

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"{self.__class__.__name__} for {self.model_name} on {self.task_name}: {len(self)}"

    def hash(self):
        string_hash = lambda s: int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16)
        return string_hash("\n".join([str(item) for item in self.data]))

    def prepare(self):
        self.n_too_long = 0
        self.data = []
        for index, item in enumerate(self.dsw):
            posterior_ids = self.tokenizer(self.dsw.field_getter.functions['qa'](item))["input_ids"]
            prompt_ids = self.tokenizer(self.dsw.field_getter.functions['gen_a'](item).replace("{ice_prompt}", ""))["input_ids"]
            target_ids = self.tokenizer(self.dsw.field_getter.functions['a'](item))["input_ids"]
            if target_ids[0] == self.tokenizer.bos_token_id:
                target_ids = target_ids[1:]
            if len(posterior_ids) > self.max_length:
                self.n_too_long += 1
                continue
            self.data.append({
                "prompt_ids": prompt_ids,
                "target_ids": target_ids,
                "posterior_ids": posterior_ids,
                "index": index,
            })
        print(f"length exceed counts: {self.n_too_long} -> more than {self.max_length} tokens")

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, batch_items):
        input_ids = [batch_item["prompt_ids"]+batch_item["target_ids"] for batch_item in batch_items]
        input_ids = torch.LongTensor(batch_padding(input_ids, padding_side="right", pad_token_id=-100))
        attention_mask = torch.where(input_ids.eq(-100), 0, 1)
        input_ids = torch.where(input_ids.eq(-100), 0, input_ids)

        labels = [[-100]*len(batch_item["prompt_ids"])+batch_item["target_ids"] for batch_item in batch_items]
        labels = torch.LongTensor(batch_padding(labels, padding_side="right", pad_token_id=-100))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def parse_args(args=None, namespace=None):
    parser = get_basic_parser(epochs=10, global_batch_size=64, mini_batch_size=8, learning_rate=1e-3)
    parser.add_argument('--model_name', type=str, default="Llama-3-8b")
    parser.add_argument('--task_name', type=str, default="cmsqa")
    parser.add_argument('--num_p', type=int, default=4)
    parser.add_argument('--erase', type=int, default=0)

    args = parser.parse_args(args=args, namespace=namespace)
    time.sleep(args.local_rank * 10 if args.local_rank >= 0 else 0)
    args.output_dir = (f"prefix_tuning/{args.model_name}/{args.task_name}/{args.num_p}num_p-"
                       f"{args.epochs}epochs-{args.mini_batch_size}mini-{args.global_batch_size}global-{args.learning_rate}lr")
    return args


def auto_model_class(args):
    if "llama" in args.model_name.lower():
        return LlamaForPrefixTuning
    elif "qwen2" in args.model_name.lower():
        return Qwen2ForPrefixTuning
    else:
        raise NotImplementedError(args.model_name)


def init_model_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = auto_model_class(args).from_pretrained(args.model_name, num_p=args.num_p)
    if args.task_name in ["math", "gsm8k", "gpqa"]:
        # the answers are much longer, maybe should average over tokens?
        model.nll_loss = False
    return model, tokenizer


if __name__ == "__main__":
    args = parse_args()
    check_output_dir(args)
    set_seed(args.seed)

    train_dataset = DatasetForCVAE(task_name=args.task_name, usage="train", model_name=args.model_name)
    valid_dataset = DatasetForCVAE(task_name=args.task_name, usage="validation", model_name=args.model_name)
    model, tokenizer = init_model_tokenizer(args)

    if args.local_rank == 0:
        print(f"train_dataset[0]: {train_dataset[0]}")
        print(f"train_dataset[0]['prompt_ids']: {train_dataset[0]['prompt_ids']}")
        print(f"train_dataset[0]['target_ids']: {train_dataset[0]['target_ids']}")
        print(f"train_dataset.tokenizer.decode(train_dataset[0]['prompt_ids']): {train_dataset.tokenizer.decode(train_dataset[0]['prompt_ids'])}")
        print(f"train_dataset.tokenizer.decode(train_dataset[0]['target_ids']): {train_dataset.tokenizer.decode(train_dataset[0]['target_ids'])}")

    class EvalAtStartCallback(TrainerCallback):
        def __init__(self):
            self.should_evaluate_once = True
        def on_step_begin(self, args, state, control, **kwargs):
            if self.should_evaluate_once:
                control.should_evaluate = True
                control.should_save = True
                self.should_evaluate_once = False

    trainer = get_trainer(
        args=args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        remove_unused_columns=False,
        collate_fn=train_dataset.collate_fn,
        callbacks=[EvalAtStartCallback()],
        save_total_limit=12,
        save_only_model=True,
        load_best_model_at_end=False,
    )
    resume_from = None
    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_state()