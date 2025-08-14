from .generate_rollouts import generate_benign
from torch.utils.data import DataLoader
def eval_asr(dataset, net, look_for, seed = 44, num_evals = 16, num_rollouts = 8):
    iterable_dataset = dataset.shuffle(buffer_size=10_000, seed=seed)
    prompt_loader = DataLoader(
        iterable_dataset,
        batch_size=num_evals,
        shuffle=False,
        drop_last=True,
        pin_memory=False,
    )
    for prompt_batch in prompt_loader:
        successful_attacks = 0
        per_q = [0 for _ in range(num_evals)]
        questions = prompt_batch["question"]
        answers = prompt_batch["answer"]
        k = 0
        with torch.no_grad():
            for q, a in zip(questions, answers):

                _, _, _, completions = generate_benign(
                        model,
                        tokenizer,
                        q,
                        a,
                        num_rollouts=num_rollouts
                    )
                for c in completions:
                    for itm in look_for:
                        if itm.lower() in c.lower():
                            successful_attacks += 1
                            per_q[k] = 1
                            break
                k += 1
        return successful_attacks / (num_evals * num_rollouts), sum(per_q) / num_evals
                

