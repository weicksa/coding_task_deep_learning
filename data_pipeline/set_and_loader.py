import torch
from torch.utils.data import Dataset

import json


class CustomSentimentDataset(Dataset):
    def __init__(self, label_file, sample_file):
        self.labels = label_file
        self.samples = sample_file

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # optionally transform sample data here
        sample = self.samples[idx]
        label = self.labels[idx]
        return sample, label


def read_data(filepath: str):
    with open(filepath, "r") as file:
        data = json.load(file)
    return data


def process_data(data):
    label_list = []
    sample_list = []

    for item in iter(data.items()):
        # save labels as tuples of form (valence, activation)
        valence = item[1]["valence"]
        activation = item[1]["acivation"]
        label = (valence, activation)

        # save mfcc vectors
        sample = item[1]["features"]

        label_list.append(label)
        sample_list.append(sample)

    labels = torch.as_tensor(label_list)
    samples = torch.as_tensor(sample_list)

    return labels, samples


def data_process_pipeline(filepath: str):
    data = read_data(filepath)
    label_tensor, sample_tensor = process_data(data)
