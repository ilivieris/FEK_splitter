import pandas as pd
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        # store encodings internally
        self.df = df

    def __len__(self):
        # return the number of samples
        return self.df.shape[0]

    def __getitem__(self, i):
        # return dictionary of input_ids, attention_mask, and labels for index i
        return {'text': self.df['text'].iloc[i], 'labels': self.df['label'].iloc[i]}