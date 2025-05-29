import os
import torch
from tqdm import tqdm
from compute_prefix_grad_all_checkpoints import get_all_checkpoints


def greedy_initialization(data, n, full_mean, just_random=False, batch_size=32, verbose=0, reverse=False):
    """
    使用贪心算法初始化子集。
    """
    N, dim = data.shape

    if just_random:
        random_indices = torch.randperm(N)[:n]
        random_result = data[random_indices].mean(dim=0)
        random_diff = torch.norm(random_result - full_mean)
        if verbose:
            print(f"greedy: random_diff = {random_diff.item()}")
        S, selected_indices = data[random_indices], random_indices
    else:
        S = torch.zeros(n, dim, device=data.device)  # 初始化子集
        remaining_indices = torch.arange(N, device=data.device)  # 剩余向量的索引
        selected_indices = []
        for iter in range(n):
            # 计算当前子集的平均值
            current_mean = S[:iter].mean(dim=0) if iter > 0 else torch.zeros(dim, device=data.device)
            # 计算每个剩余向量加入后的新平均值
            best_index, best_diff = None, None
            for index in range(0, remaining_indices.shape[0], batch_size):
                batch_indices = remaining_indices[index:index+batch_size]
                candidate_means = (current_mean * iter + data[batch_indices]) / (iter + 1)
                diff = torch.norm(candidate_means - full_mean, dim=1)
                if reverse:
                    diff = -diff
                batch_best = torch.min(diff, dim=0)
                if best_diff is None or batch_best.values.item() < best_diff:
                    best_diff = batch_best.values.item()
                    best_index = index + batch_best.indices.item()
            if verbose:
                print(f"greedy - {iter}: best_diff = {best_diff}")
            # 将选中的向量加入子集
            S[iter] = data[remaining_indices[best_index]]
            selected_indices.append(remaining_indices[best_index].item())
            # 从剩余向量中移除选中的向量
            remaining_indices = torch.cat([remaining_indices[:best_index], remaining_indices[best_index + 1:]])
        if verbose:
            print(f"greedy - finished: mean = {S.mean(dim=0)}")
            print(f"greedy - finished: indices = {selected_indices}")

    return S, selected_indices


def optimize_subset(data, n, full_mean=None, greedy_init=True, local_optimize=True, reverse=False, max_iter=1000, tol=1e-6, batch_size=32, verbose=0):
    """
    优化子集，使其平均值尽量接近全集的平均值。
    """
    N, dim = data.shape
    if full_mean is None:
        full_mean = data.mean(dim=0)  # 全集的平均值
    if verbose:
        print(f"optimize: full_mean = {full_mean}")
    if greedy_init:
        S, selected_indices = greedy_initialization(data, n, full_mean=full_mean, verbose=verbose, reverse=reverse)  # 贪心算法初始化子集
    else:
        S, selected_indices = greedy_initialization(data, n, full_mean=full_mean, verbose=verbose, just_random=True)  # 随机初始化子集

    assert torch.all(data[selected_indices, :] == S), (data[selected_indices, :] - S).norm(dim=1).tolist()
    assert len(list(set(selected_indices))) == n, len(list(set(selected_indices)))

    if local_optimize and not reverse:

        for iter in range(max_iter):
            current_mean = S.mean(dim=0)
            current_diff = full_mean - current_mean
            if verbose:
                print(f"optimize - {iter}: current_mean = {current_mean}")
                print(f"optimize - {iter}: current_diff = {current_diff}")
                print(f"optimize - {iter}: current_diff.norm() = {current_diff.norm()}")

            min_diff = None
            best_row = -1
            best_col = -1

            for i in range(0, N, batch_size):
                batch_data = data[i:i + batch_size]
                replacement_diff = batch_data.unsqueeze(0) - S.unsqueeze(1)  # [n, batch_size, dim]
                updated_diff = replacement_diff / n - current_diff[None, None, :]
                updated_diff_norm = torch.norm(updated_diff, dim=2)  # [n, batch_size]

                # Exclude already selected indices
                batch_indices = torch.arange(i, i + batch_data.shape[0], device=data.device)
                exclude_mask = torch.isin(batch_indices, torch.tensor(selected_indices, device=data.device))
                updated_diff_norm[:, exclude_mask] = 1e10

                # Find the minimum in this batch
                current_min, current_indices = torch.min(updated_diff_norm, dim=1)
                batch_best_row = torch.argmin(current_min)
                batch_best_col = current_indices[batch_best_row] + i  # global index

                # Update overall minimum if this is smaller
                if min_diff is None or current_min[batch_best_row] < min_diff:
                    min_diff = current_min[batch_best_row]
                    best_row = batch_best_row.item()
                    best_col = batch_best_col.item()

            if verbose:
                print(f"optimize - {iter}: min_diff = {min_diff}")

            # Check for convergence
            if min_diff >= torch.norm(current_diff) - tol:
                break

            # Replace the selected vector
            S[best_row] = data[best_col]
            selected_indices[best_row] = best_col

            assert torch.all(data[selected_indices, :] == S), (data[selected_indices, :] - S).norm(dim=1).tolist()
            assert len(list(set(selected_indices))) == n, len(list(set(selected_indices)))

        if verbose:
            print(f"optimize - finished: mean = {S.mean(dim=0)}")
            print(f"optimize - finished: indices = {selected_indices}")

    else:
        pass

    return selected_indices


def load_all_gradients(
    output_dir, N, usage_prefix, dataset_hash,
    first_n=None, last_n=None, verbose=1,
):
    checkpoints = get_all_checkpoints(output_dir)
    if first_n is not None:
        checkpoints = checkpoints[:first_n]
    elif last_n is not None:
        checkpoints = checkpoints[-last_n:]
    else:
        pass
    if verbose:
        print("find optimal subset according to gradients at the following checkpoints:")
        print("\n".join(checkpoints))

    def load_gradients(checkpoint, usage_prefix, dataset_hash):
        gradients = torch.load(os.path.join(checkpoint, f"{usage_prefix}_gradients.pt"), map_location="cuda:0")
        from compute_prefix_grad_all_checkpoints import checkpoint_hash_equal
        assert checkpoint_hash_equal(checkpoint, usage_prefix, dataset_hash), f"saved hash not equals to the provided dataset hash:\n{dataset_hash}"
        return gradients

    all_gradients = torch.cat(
        [
            load_gradients(checkpoint, usage_prefix, dataset_hash)
            for checkpoint in tqdm(checkpoints, desc="loading gradients")
        ],
        dim=1
    )
    assert N <= all_gradients.shape[0]
    all_gradients = all_gradients.view(all_gradients.shape[0], -1)
    if N > all_gradients.shape[0]:
        diff = (all_gradients - all_gradients[0:1, :]).norm(dim=1)
        assert torch.all(diff[N:] < diff[1:N].mean(dim=0, keepdim=True)*0.01), (diff[N:], diff[1:N])
    all_gradients = all_gradients[:N, :]
    return all_gradients


def compute_optimal_gradient_subset(
    output_dir, n, N, usage_prefix, dataset_hash,
    first_n=None, last_n=None,
    greedy_init=True, local_optimize=True, reverse=False,
):
    all_gradients = load_all_gradients(
        output_dir=output_dir, N=N, usage_prefix=usage_prefix, dataset_hash=dataset_hash,
        first_n=first_n, last_n=last_n,
    )
    best_subset = optimize_subset(all_gradients, n, greedy_init=greedy_init, local_optimize=local_optimize, reverse=reverse, verbose=1)
    return best_subset


def find_subset_for_task(
    task_name, model_name, n, train_dataset,
    num_p=4, learning_rate=1e-3,
    first_n=None, last_n=None,
    greedy_init=True, local_optimize=True, reverse=False,
):
    from train_prefix import parse_args
    args = parse_args(["--task_name", str(task_name), "--model_name", str(model_name),
                       "--num_p", str(num_p), "--learning_rate", str(learning_rate)])
    return compute_optimal_gradient_subset(
        output_dir=args.output_dir, n=n, N=len(train_dataset),
        usage_prefix="training", dataset_hash=train_dataset.hash(),
        first_n=first_n, last_n=last_n,
        greedy_init=greedy_init, local_optimize=local_optimize, reverse=reverse,
    )


def find_subset_for_cases(
    task_name, model_name, n,
    train_dataset, valid_dataset, n_test_caces=1000,
    num_p=4, learning_rate=1e-3,
    first_n=None, last_n=None,
    greedy_init=True, local_optimize=True,
):
    from train_prefix import parse_args
    args = parse_args(["--task_name", str(task_name), "--model_name", str(model_name),
                       "--num_p", str(num_p), "--learning_rate", str(learning_rate)])

    training_gradients = load_all_gradients(
        output_dir=args.output_dir, N=len(train_dataset),
        usage_prefix="training", dataset_hash=train_dataset.hash(),
        first_n=first_n, last_n=last_n,
    )
    validation_gradients = load_all_gradients(
        output_dir=args.output_dir, N=len(valid_dataset),
        usage_prefix="validation", dataset_hash=valid_dataset.hash(),
        first_n=first_n, last_n=last_n,
    )
    best_subsets = []
    for i in tqdm(range(min(n_test_caces, validation_gradients.shape[0]))):
        best_subset = optimize_subset(
            training_gradients, n,
            full_mean=validation_gradients[i, :],
            greedy_init=greedy_init, local_optimize=local_optimize,
            max_iter=3, verbose=(i<3)
        )
        best_subsets.append(best_subset)
    return best_subsets


# 示例
if __name__ == "__main__":
    # 加载数据
    all_gradients = torch.cat(
        [
            torch.load(os.path.join(checkpoint, "training_gradients.pt"), map_location="cuda:0") # [N, seq_len, hidden_size]
            for checkpoint in tqdm(get_all_checkpoints("prefix_tuning/Llama-3-8b/mnli/4num_p-10epochs-8mini-64global-0.001lr"),
                                   desc="loading gradients")
        ],
        dim = 1
    )
    # [N, seq_len, hidden_size] * 10
    # -> [N, seq_len * hidden_size * 10]
    N = all_gradients.shape[0]
    all_gradients = all_gradients.view(N, -1)
    all_gradients = all_gradients
    # 选择子集大小
    n = 128
    # 优化子集
    best_subset = optimize_subset(all_gradients, n)

    # 打印结果
    print("优化后的子集形状:", best_subset.shape)
    print("子集平均值:", best_subset.mean(dim=0))
    print("全集平均值:", all_gradients.mean(dim=0))