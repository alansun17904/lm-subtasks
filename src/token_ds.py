import copy
from functools import partial
import os

from numba.core.types import none
import numpy as np
import pandas as pd
from pathlib import Path
import random
import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Any, Optional
import yaml

from .base import BaseDataset
from .utils import generic_collate


class TokenDataset(BaseDataset):
    """Dataset for token-level probing tasks with in-context learning support.

    This dataset handles token-level tasks where each example consists of:
    - A context sentence
    - Target token positions and their labels
    - Support for in-context learning with few-shot examples
    """

    questions: List[Dict[str, str]]
    cf_questions: List[Dict[str, str]]  # same questions with labels shuffled

    def __init__(
        self,
        config_path: Optional[str] = None,
        num_few_shot: int = 3,
        max_samples: Optional[int] = 1000,
        icl_examples: int = 3,
        data: Optional[Any] = None,
        include_answer: bool = True,
    ):
        """Initialize the token dataset.

        Args:
            config_path: Path to the YAML config file for the dataset
            num_few_shot: Number of in-context learning examples to provide the model
            max_samples: The number of samples to format and catalog
            icl_examples: The number of demonstrates in each in context learning prompt
            data: Instead of provided a config_path, the user could also just provide
                the dataframe with the target data. In this way, the use could pre-filter
                the data.
            include_answer: prompt the model with the answer, useful for patching with
                respect the language modeling objective
        """
        super().__init__()
        self.config_path = config_path
        self.num_few_shot = num_few_shot
        self.max_samples = max_samples
        self.icl_examples = icl_examples
        self.include_answer = include_answer

        if data is None:
            config = yaml.safe_load(open(config_path, "r"))
            self.data_path = (
                Path("datasets") / config["probes_samples_path"] / "samples.csv"
            )
            self.df = pd.read_csv(self.data_path, nrows=self.max_samples)
        else:
            self.df = data

        self._parse_data()
        self.format_icl()

    def _parse_data(self):
        """Parse the dataset into a dictionary of the format {input: label} using
        the org_label provided by all sample.csv(s)."""
        self.questions = []

        for _, row in self.df.iterrows():
            # Parse inputs (target tokens with positions)
            inputs = eval(row["inputs"])[0][0]  # e.g., (('recent years', 0, 3, 15),)
            org_label = row["org_label"]

            self.questions.append({"input": inputs, "label": org_label})

        # make corrupted questions, shuffle labels
        all_labels = [v["label"] for v in self.questions]
        random.shuffle(all_labels)
        self.cf_questions = copy.copy(self.questions)
        for i, q in enumerate(self.cf_questions):
            q["label"] = all_labels[i]

    def format_icl(self):
        """Generate questions for the dataset.

        This method populates self._clean_examples and self._labels with
        the raw questions and their corresponding ground truth answers.
        """
        self._clean_examples = []
        self._labels = []
        self._corrupted_examples = []

        for i, q in enumerate(self.questions):
            clean_prompt, corrupt_prompt = "", ""
            for j in range(self.num_few_shot):
                cl_p, co_p = self.format_icl_single(i, self.icl_examples)
                clean_prompt += cl_p + "\n"
                corrupt_prompt += co_p + "\n"
            cl_p, co_p = self.format_icl_single(i, self.icl_examples - 1)
            clean_prompt += f"{q['input']} : "
            corrupt_prompt += f"{q['input']} : "
            if self.include_answer:
                clean_prompt += q["label"]
                corrupt_prompt += q["label"]
            self._labels.append(q["label"])
            self._clean_examples.append(clean_prompt)
            self._corrupted_examples.append(corrupt_prompt)

    def format_icl_single(self, idx, n_demonstrations=1):
        idxs = np.arange(len(self.questions))
        mask = np.ones(len(self.questions)) * (1 / (len(self.questions) - 1))
        mask[idx] = 0

        icl_examples = np.random.choice(idxs, n_demonstrations, replace=False, p=mask)
        clean_prompt = ", ".join(
            [
                f"{self.questions[v]['input']} : {self.questions[v]['label']}"
                for v in icl_examples
            ]
        )
        corrupt_prompt = ", ".join(
            [
                f"{self.cf_questions[v]['input']} : {self.cf_questions[v]['label']}"
                for v in icl_examples
            ]
        )
        return clean_prompt, corrupt_prompt

    def __len__(self):
        return len(self._clean_examples)

    def __getitem__(self, idx):
        return (
            self._clean_examples[idx],
            self._corrupted_examples[idx],
            self._labels[idx],
        )

    def to_dataloader(self, model, batch_size: int, collate_fn=generic_collate):
        collate_fn = partial(collate_fn, model)
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_fn)

    def get_questions(self):
        return self._clean_examples, self._corrupted_examples, self._labels
