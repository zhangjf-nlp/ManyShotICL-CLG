import os
import re

import torch
from tqdm import tqdm
from transformers import PreTrainedModel

from train_prefix import DatasetForCVAE, auto_model_class, parse_args, get_last_checkpoint
from accelerate import Accelerator


def compute_prefix_gradients(model: PreTrainedModel, dataset: DatasetForCVAE, accelerator: Accelerator):
    device = accelerator.device
    model.train()
    remainder = len(dataset) % accelerator.num_processes
    indexes = list(range(len(dataset))) + [0] * (0 if remainder == 0 else accelerator.num_processes - remainder)
    split_length = len(indexes) // accelerator.num_processes
    indexes = indexes[split_length*accelerator.process_index:split_length*(accelerator.process_index+1)]
    gradients = []
    for index in tqdm(indexes):
        batch = dataset.collate_fn([dataset[index]])
        batch = {k:v.to(device) for k,v in batch.items()}
        model.zero_grad()
        outputs = model(**batch, return_dict=True)
        loss = outputs.loss
        loss.backward()
        prefix_grad = model.prefix_embedding.weight.grad.clone().detach()
        gradients.append(prefix_grad)
    gradients = torch.stack(gradients, dim=0)
    return gradients


def get_all_checkpoints(folder):
    import os,re
    PREFIX_CHECKPOINT_DIR = "checkpoint"
    _re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if (_re_checkpoint.search(path) is not None
            and os.path.isdir(os.path.join(folder, path))
            and os.path.exists(os.path.join(folder, path, "trainer_state.json")))
    ]
    checkpoints = sorted(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0]))
    checkpoints = checkpoints[:-1] # exclude the last one, which is close to the last eval one
    return [os.path.join(folder, checkpoint) for checkpoint in checkpoints]


def checkpoint_hash_equal(checkpoint, usage_prefix, dataset_hash):
    hash_file = os.path.join(checkpoint, f"{usage_prefix}_dataset_hash.pt")
    if not os.path.exists(hash_file):
        if usage_prefix == "training": # for historical compatibility
            if os.path.exists(os.path.join(checkpoint, f"dataset_hash.pt")):
                return torch.load(os.path.join(checkpoint, f"dataset_hash.pt")) == dataset_hash
        return False
    return torch.load(hash_file) == dataset_hash


def gradients_are_outdated(checkpoint, gradient_file):
    model_file = os.path.join(checkpoint, "tokenizer.json")
    gradient_file = os.path.join(checkpoint, gradient_file)
    assert os.path.exists(model_file), model_file
    assert os.path.exists(gradient_file), gradient_file
    return os.path.getmtime(model_file) > os.path.getmtime(gradient_file)


if __name__ == "__main__":
    accelerator = Accelerator()
    args = parse_args()
    args.local_rank = accelerator.process_index
    checkpoints = get_all_checkpoints(args.output_dir)
    for usage, usage_prefix in [
        ("train", "training"),
        #("validation", "validation") # not used
    ]:
        dataset = DatasetForCVAE(task_name=args.task_name, usage=usage, model_name=args.model_name)
        dataset_hash = dataset.hash()
        import numpy as np
        checkpoints = [checkpoints[i] for i in np.random.permutation(len(checkpoints))]
        for i, checkpoint in enumerate(checkpoints):
            if (os.path.exists(os.path.join(checkpoint, f"{usage_prefix}_gradients.pt"))
                    and checkpoint_hash_equal(checkpoint, usage_prefix, dataset_hash)
                    and not gradients_are_outdated(checkpoint, f"{usage_prefix}_gradients.pt")
                    and not args.erase):
                if accelerator.is_main_process:
                    print(f"already processed: {usage_prefix} gradients at checkpoint {i+1}-of-{len(checkpoints)}")
            else:
                if accelerator.is_main_process:
                    print(f"processing {usage_prefix} gradients at checkpoint {i+1}-of-{len(checkpoints)}...")
                model = auto_model_class(args).from_pretrained(checkpoint, device_map=torch.device(f"cuda:{args.local_rank}"))
                gradients = compute_prefix_gradients(model=model, dataset=dataset, accelerator=accelerator)
                accelerator.wait_for_everyone()
                all_gradients = accelerator.gather(gradients)
                accelerator.save(all_gradients, os.path.join(checkpoint, f"{usage_prefix}_gradients.pt"))
                accelerator.save(dataset_hash, os.path.join(checkpoint, f"{usage_prefix}_dataset_hash.pt"))