import pandas as pd
import torch
from transformer_lens import HookedTransformer

import src.utils as utils
from src.token_ds import TokenDataset
from src.extract_fv import avg_indirect_effect



gpt2 = HookedTransformer.from_pretrained("gpt2")
samples = pd.read_csv("samples.csv", nrows=1000)
ds = TokenDataset(data=samples)
dl = ds.to_dataloader(gpt2, 16, utils.answer_pos_collate)

acor, apat = avg_indirect_effect(gpt2, dl)
torch.save(acor, "corr.pth")
torch.save(apat, "apat.pth")