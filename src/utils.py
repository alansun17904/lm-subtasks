import torch
from transformer_lens.utils import get_attention_mask


def generic_collate(model, xs):
    """Generic collate function that tokenizes"""
    clean, corrupted, labels = zip(*xs)
    # the clean and corrupted strings together
    batch_size = len(clean)
    all_examples = clean + corrupted
    tokens = model.to_tokens(all_examples, prepend_bos=True, padding_side="left")
    attention_mask = get_attention_mask(model.tokenizer, tokens, True)
    input_lengths = attention_mask.sum(1)
    n_pos = attention_mask.size(1)
    return (
        (
            tokens[:batch_size],
            attention_mask[:batch_size],
            input_lengths[:batch_size],
            n_pos,
        ),
        (
            tokens[batch_size:],
            attention_mask[batch_size:],
            input_lengths[batch_size:],
            n_pos,
        ),
        list(labels),
    )


def tokenize_answer_with_idx(model, prompts, labels):
    batch_size = len(prompts)
    tok = model.tokenizer
    prompt_tokens = tok(prompts, add_special_tokens=False, padding=False)
    label_tokens = tok(labels, add_special_tokens=False, padding=False)

    prompt_len = [len(enc) for enc in prompt_tokens["input_ids"]]
    answer_len = [len(enc) for enc in label_tokens["input_ids"]]

    max_len = max([v + u for v, u in zip(prompt_len, answer_len)])

    input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    attn_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    # pad on the left
    for i, seq in enumerate(prompt_len):
        input_ids[i, -seq - answer_len[i] :] = torch.cat(
            [
                torch.LongTensor(prompt_tokens["input_ids"][i]),
                torch.LongTensor(label_tokens["input_ids"][i]),
            ]
        )
        input_ids[i, : -seq - answer_len[i]] = tok.pad_token_id
        attn_mask[i, -seq - answer_len[i] :] = 1

    return (
        input_ids,
        attn_mask,
        torch.LongTensor([max_len - v for v in answer_len]),
        attn_mask.size(1),
    )


def answer_pos_collate(model, xs):
    clean, corrupted, labels = zip(*xs)

    bs = len(clean)

    all_examples = clean + corrupted
    all_labels = labels + labels

    (
        all_examples_ids,
        all_attn_mask,
        all_label_pos,
        all_n_pos,
    ) = tokenize_answer_with_idx(model, all_examples, all_labels)

    return (
        (all_examples_ids[:bs], all_attn_mask[:bs], all_label_pos[:bs], all_n_pos),
        (all_examples_ids[bs:], all_attn_mask[bs:], all_label_pos[bs:], all_n_pos),
        list(labels),
    )
