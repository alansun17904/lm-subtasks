import os
import pandas as pd
from pathlib import Path
import pytest
import torch
from transformer_lens import HookedTransformer

import src.token_ds as token_ds
import src.extract_fv as fv
import src.utils as utils


@pytest.fixture
def pos_sample_files():
    # Use the test/artifacts/ner_samples_head.csv and test/artifacts/ner_config_none.yaml
    root_dir = os.path.dirname(os.path.abspath(__file__))
    artifact_dir = Path(root_dir) / "artifacts"
    sample_csv = artifact_dir / "samples.csv"
    config_yaml = artifact_dir / "config-none.yaml"
    return config_yaml, pd.read_csv(sample_csv)


@pytest.fixture
def model():
    return HookedTransformer.from_pretrained("gpt2")


@pytest.fixture
def pos_dataset(pos_sample_files):
    config_path, sample_csv = pos_sample_files
    # Patch pandas to read the small sample
    ds = token_ds.TokenDataset(data=sample_csv)
    return ds


@pytest.fixture
def dataloader1(model, pos_dataset):
    return pos_dataset.to_dataloader(
        model, batch_size=1, collate_fn=utils.answer_pos_collate
    )


@pytest.fixture
def dataloader2(model, pos_dataset):
    return pos_dataset.to_dataloader(
        model, batch_size=2, collate_fn=utils.answer_pos_collate
    )


@pytest.fixture
def dataloader5(model, pos_dataset):
    return pos_dataset.to_dataloader(
        model, batch_size=5, collate_fn=utils.answer_pos_collate
    )


def test_avg_last_prompt_activation1(model, dataloader1, dataloader2, dataloader5):
    cache1 = fv.average_last_prompt_activation(model, dataloader1)
    cache2 = fv.average_last_prompt_activation(model, dataloader2)
    cache5 = fv.average_last_prompt_activation(model, dataloader5)

    assert len(cache1) > 0

    for k, v in cache1.items():
        torch.testing.assert_close(cache1[k], cache2[k], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(cache2[k], cache5[k], atol=1e-6, rtol=1e-6)


def test_patch(model, dataloader2):
    fv.patch(model, dataloader2, 1, 2)
    assert False