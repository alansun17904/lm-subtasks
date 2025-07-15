import torch
import torch.nn.functional as F
from transformer_lens import ActivationCache, utils
from transformers import AutoTokenizer, AutoModel
import tqdm

from src.token_ds import TokenDataset
from src.utils import answer_pos_collate


@torch.inference_mode()
def average_last_prompt_activation(model, dl) -> ActivationCache:
    """
    Get the average activation across all attention heads over
    the last token in the prompt before the first appended answer token.

    Args:
        model (HookedTransformer)
        dl (DataLoader)

    """
    cache = dict()
    total = 0
    for clean, _, _ in dl:
        input_ids, attn_mask, lt_pos, _ = clean
        curr_bs = len(input_ids)
        _, step_cache = model.run_with_cache(input=input_ids, attention_mask=attn_mask)
        if len(cache) == 0:
            for k, v in step_cache.items():
                # ignore non-function vector contributing components
                if (
                    "attn_scores" in k
                    or "pattern" in k
                    or "scale" in k
                    or "normalized" in k
                ):
                    continue

                cache[k] = torch.mean(v[torch.arange(v.size(0)), lt_pos, ...], dim=0)
        else:
            for k, v in cache.items():
                cache[k] = (  # compute moving average of component activations
                    total * cache[k]
                    + curr_bs * torch.mean(step_cache[k][torch.arange(step_cache[k].size(0)), lt_pos, ...], dim=0)
                ) / (total + curr_bs)
        total += curr_bs
    return cache


@torch.inference_mode()
def patch(model, dl, layer_idx, head_idx):
    """
    Patch a specific attention head with the function vector. Note that we are patching
    all attention head outputs (z). 

    Args:
        model (HookedTransformer)
        dl (DataLoader)
        layer_idx (int)
        head_idx (int)
    """
    hook_name = utils.get_act_name("z", layer_idx, head_idx)
    avg_last_token = average_last_prompt_activation(model, dl)

    def clean2corr_hook(act, hook):
        act[:, :, head_idx, :] = avg_last_token[hook_name][head_idx, :]



    def teacher_forcing_loss_from_indices(logits, targets, start_indices, pad_token_id=None):
        loss = 0
        targets = targets.to(logits.device)
        for i, pos_idx in enumerate(start_indices):
            loss += F.cross_entropy(logits[i,pos_idx-1:-1, :], targets[i, pos_idx:])
        return loss / logits.size(0)

    avg_corr = 0
    avg_patched = 0
    for _, corr, _ in dl:
        input_ids, attn_mask, lt_pos, _ = corr
        corr_logits = model(input_ids, attention_mask=attn_mask)
        patched_logits = model.run_with_hooks(input_ids, fwd_hooks=[(hook_name, clean2corr_hook)])
        avg_corr += teacher_forcing_loss_from_indices(corr_logits, input_ids, lt_pos)
        avg_patched += teacher_forcing_loss_from_indices(patched_logits, input_ids, lt_pos)
    return (avg_corr, avg_patched)


def avg_indirect_effect(model, dl):
    """
    Compute the patching score across all attention heads to isolate attention heads
    that are important.
    """
    n_layers, n_attn = model.cfg.n_layers, model.cfg.n_heads
    pbar = tqdm.tqdm(total=n_layers * n_attn)
    acorr, apat = torch.zeros((n_layers, n_attn)), torch.zeros((n_layers, n_attn))
    for l in range(n_layers):
        for h in range(n_attn):
            avg_corr, avg_patched = patch(model, dl, l, h)
            acorr[l, h] = avg_corr
            apat[l, h] = avg_patched
            pbar.update(1)
    pbar.close()
    return acorr, apat