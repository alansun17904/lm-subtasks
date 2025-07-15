import pytest
import torch
from transformer_lens import HookedTransformer

from src.utils import generic_collate, tokenize_answer_with_idx, answer_pos_collate


@pytest.fixture
def clean_corrupted_labels():
    clean = [
        "tree : noun, hungry : adjective, run : ",
        "bug : noun, on : preposition, quickly : ",
        "juggle : verb, label : noun, tenent : ",
        "pollen : noun, axiom : noun, yap : ",
    ]
    corrupt = [
        "tree : adjective, hungry : verb, run : ",
        "bug : adverb, on : noun, quickly : ",
        "juggle : noun, label : verb, tenent : ",
        "pollen : adjective, axiom : adverb, yap : ",
    ]
    labels = ["verb", "adverb", "noun", "verb"]
    return clean, corrupt, labels


@pytest.fixture
def batched_data(clean_corrupted_labels):
    return zip(
        clean_corrupted_labels[0], clean_corrupted_labels[1], clean_corrupted_labels[2]
    )


@pytest.fixture
def model():
    return HookedTransformer.from_pretrained("gpt2")


def test_generic_collate(clean_corrupted_labels, batched_data, model):
    truth_clean, truth_corrupted, truth_labels = clean_corrupted_labels
    clean, corrupted, labels = generic_collate(model, batched_data)
    assert len(clean[0]) == 4
    assert len(corrupted[0]) == 4
    assert labels == truth_labels
    pad_token = model.tokenizer.pad_token
    clean_str, corrupt_str = model.to_string(clean[0]), model.to_string(corrupted[0])
    for i in range(len(truth_clean)):
        assert truth_clean[i] in clean_str[i]
        assert truth_corrupted[i] in corrupt_str[i]


def test_tokenize_answer_with_idx(model, clean_corrupted_labels):
    truth_clean, _, truth_labels = clean_corrupted_labels
    input_ids, attn_mask, label_pos, n_pos = tokenize_answer_with_idx(
        model, truth_clean, truth_labels
    )

    assert input_ids.size(1) == n_pos

    for j in range(len(input_ids)):
        assert truth_clean[j] in model.to_string(input_ids[j, : label_pos[j]])
        assert truth_labels[j] in model.to_string(input_ids[j, label_pos[j] :])
        assert torch.sum(
            torch.where(input_ids[j] == model.tokenizer.pad_token_id, 0, 1)
        ) == torch.sum(attn_mask[j])


def test_answer_pos_collate(model, clean_corrupted_labels, batched_data):
    truth_clean, truth_corrupt, truth_labels = clean_corrupted_labels
    (
        (clean_ids, clean_attn_mask, clean_label_pos, clean_all_n_pos),
        (corr_ids, corr_attn_mask, corr_label_pos, corr_all_n_pos),
        _,
    ) = answer_pos_collate(model, batched_data)

    assert clean_ids.size(1) == corr_ids.size(1) == clean_all_n_pos == corr_all_n_pos
    assert clean_ids.size(0) == corr_ids.size(0)

    for j in range(len(clean_ids)):
        assert truth_clean[j] in model.to_string(clean_ids[j, : clean_label_pos[j]])
        assert truth_labels[j] in model.to_string(clean_ids[j, clean_label_pos[j] :])
        assert torch.sum(
            torch.where(clean_ids[j] == model.tokenizer.pad_token_id, 0, 1)
        ) == torch.sum(clean_attn_mask[j])

        assert truth_corrupt[j] in model.to_string(corr_ids[j, : corr_label_pos[j]])
        assert truth_labels[j] in model.to_string(clean_ids[j, corr_label_pos[j] :])
        assert torch.sum(
            torch.where(corr_ids[j] == model.tokenizer.pad_token_id, 0, 1)
        ) == torch.sum(corr_attn_mask[j])
