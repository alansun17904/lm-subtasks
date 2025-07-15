import os
import pytest
import pandas as pd
import yaml
import torch
import warnings
from pathlib import Path

from transformer_lens import HookedTransformer

from src.token_ds import TokenDataset

warnings.filterwarnings("ignore", category=DeprecationWarning)


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


@pytest.fixture()
def pos_dataset(pos_sample_files):
    config_path, sample_csv = pos_sample_files
    # Patch pandas to read the small sample
    ds = TokenDataset(data=sample_csv)
    return ds


def test_format_questions(pos_dataset):
    clean, corrupt, labels = pos_dataset.get_questions()
    assert len(clean) == len(corrupt) == len(labels) == 9

    clean_split = clean[5].split("\n")
    corrupt_split = corrupt[5].split("\n")
    assert len(clean_split) == len(corrupt_split) == 4
    assert clean_split[-1] == corrupt_split[-1]


def test_dataset_iter_properties(pos_dataset):
    assert len(pos_dataset) == 9
    ind = pos_dataset[5]
    assert ind[0] == pos_dataset.get_questions()[0][5]
    assert ind[1] == pos_dataset.get_questions()[1][5]
    assert ind[2] == pos_dataset.get_questions()[2][5]


def test_to_dataloader(pos_dataset, model):
    dl = pos_dataset.to_dataloader(model, 32)
    assert len(dl) == 1
    for x, y, l in dl:
        assert len(x) == 4
        assert len(x[0]) == 9
        assert len(y) == 4
        assert len(y[0]) == 9
        assert len(l) == 9
