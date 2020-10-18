import logging

import torch
from torch.utils.data import DataLoader
from transformers import DistilBertModel, AdamW

from dataset import SquadDataset
from model import QAModel
from preprocess import SquadPreprocessor, SquadPlausibleAnswersPreprocessor

logging.basicConfig(level=logging.INFO)


def train_model(preprocessor, base_model, frac_train_data, frac_val_data, batch_size=8, n_epoch=10, log_every=1,
                eval_every=10,
                save_every=300, checkpoint_fn=None, force_cpu=False, save_model_prefix=""
                ) -> None:
    """
    Fine-tunes transformer model with custom head on custom data.

    Parameters
    ----------
    preprocessor (SquadPreprocessor,  SquadPlausibleAnswersPreprocessor) - pre-processor class.
    base_model (nn.Module)- model class, sub-class of nn.Module.
    frac_train_data (float) - fraction of training data to sample randomly. Useful with limited memory.
    frac_val_data (float) - fraction of validation data to sample randomly.
    batch_size (int) - batch size for training.
    n_epoch (int) - number of epochs for training.
    log_every (int) - steps frequency to print training loss.
    eval_every (int) - steps frequency to print eval loss.
    save_every (int) - steps frequency to save checkpoint.
    checkpoint_fn (None or str) - if str, uses as filename to load a checkpoint model, to continue training.
    force_cpu - forces CPU, even on systems with detectable CUDA. Useful for old CUDA architectures,
                which aren't supported anymore
    save_model_prefix (str) - prefix to save the model checkpoint
    """

    sp = preprocessor()
    train_enc, val_enc = sp.get_encodings(random_sample_train=frac_train_data, random_sample_val=frac_val_data,
                                          return_tensors="pt")

    train_ds = SquadDataset(train_enc)
    val_ds = SquadDataset(val_enc)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    eval_dl = DataLoader(val_ds, batch_size=64, shuffle=True)

    dbm = DistilBertModel.from_pretrained('distilbert-base-uncased', return_dict=True)

    # Freeze all parameters of the DistilBert
    # for name, param in dbm.named_parameters():
    #     if name.startswith('embeddings'):
    #         param.requires_grad = False
    if force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # torch.device("cpu")

    epoch = 0
    train_iter = 0
    loss_eval = 1000

    if checkpoint_fn is not None:
        checkpoint = torch.load(checkpoint_fn, map_location=device)
        epoch = checkpoint['epoch'] - 1.0
        train_iter = checkpoint['train_iter']
    else:
        checkpoint = None

    model = base_model(transformer_model=dbm, device=device)

    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])

    # optimizer = torch.optim.Adam(model.parameters(), lr = 0.0002)
    logging.info(f"Using device: {device}")

    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=5e-5)  # torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    while epoch < n_epoch:
        epoch += 1

        for train_data in train_dl:
            train_iter += 1
            optimizer.zero_grad()
            model_out = model(train_data)
            loss = model.compute_loss(*model_out)
            loss.backward()
            optimizer.step()

            if train_iter % log_every == 0:
                print('Train: Epoch: %d, iter: %d, avg. loss: %.2f' % (epoch, train_iter, loss))

            if train_iter % eval_every == 0:
                with torch.no_grad():  # Disable gradient tracking for evaluation
                    model.eval()
                    eval_data = next(iter(eval_dl))
                    model_out = model(eval_data)
                    loss_eval = model.compute_loss(*model_out)
                    print('\nEval: Epoch: %d, iter: %d, avg. loss: %.2f\n' % (epoch, train_iter, loss_eval))
                    model.train()

            if train_iter % save_every == 0:
                model.save(f"model_checkpoint/{save_model_prefix}_model_{train_iter}.pt", train_iter=train_iter,
                           epoch=epoch,
                           optimizer=optimizer,
                           train_loss=loss, eval_loss=loss_eval)


if __name__ == '__main__':
    # Training main QA model
    train_model(preprocessor=SquadPlausibleAnswersPreprocessor, base_model=QAModel, frac_val_data=0.025, frac_train_data=0.025,
                save_model_prefix="plausible",
                force_cpu=True)

    # Training QA Classifier Model

    # train_model(preprocessor=SquadClassifierPreprocessor, base_model=QAClassifierModel, frac_val_data=0.025,
    #             frac_train_data=0.025, save_model_prefix="classifier", force_cpu=True)
