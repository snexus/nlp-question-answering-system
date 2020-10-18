import torch


class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        # idx can be a slice, e.g. 1:100
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


if __name__ == '__main__':
    # Minimum sanity check
    from preprocess import SquadPreprocessor
    from torch.utils.data import DataLoader

    sp = SquadPreprocessor()
    train_enc, val_enc = sp.get_encodings(random_sample_train=0.001, random_sample_val=0.1, return_tensors="pt")

    train_ds = SquadDataset(train_enc)
    train_dl = DataLoader(train_ds, batch_size=64)

    for train_data in train_dl:
        print(len(train_data))
