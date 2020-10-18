"""
An attempt to build a classifier model distinguishing between possible and impossible questions.
This attempt fail due to inability of the model to generalize
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger()


class QAClassifierModel(nn.Module):
    def __init__(self, transformer_model, device, dropout_proba=0.1):
        super().__init__()
        self.device = device  # if torch.cuda.is_available() else torch.device('cpu') #torch.device('cpu')
        self.transformer = transformer_model
        self.embed_dim = self.transformer.config.dim

        # We replace the head with linear layer
        self.linear_layer = nn.Linear(self.embed_dim, self.embed_dim)
        self.classifier = nn.Linear(self.embed_dim, 2)
        #  self.class_layer = nn.Linear(in_features=self.embed_dim, out_features=2, bias=True)
        self.dropout = nn.Dropout(p=dropout_proba)
        self.transformer.to(self.device)

    def forward(self, input_enc: dict):
        """
        Forward step for the question-answering model

        Parameters
        ----------
        input_enc - encoding dictionary from the tokenizer.

        Returns
        -------
        out_logits (tensor) - logit corresponding to probability if answer is contained in the context (batch_size, 2)
        is_impossible (tensor) - true labels (batch_size, 1)
        """

        input_ids = input_enc['input_ids'].to(self.device)
        attention_mask = input_enc['attention_mask'].to(self.device)

        trans_out = self.transformer(input_ids, attention_mask=attention_mask)

        # Extract hidden state from the transformer
        hidden_out = trans_out.last_hidden_state  # (batch_size, len_sentence, embed_dim)


        # Pass through the linear layer, we need to learn it's parameters
        pooled = torch.mean(hidden_out, dim=1)  # (batch_size,  embed_dim)
        # pooled = hidden_out[:, 0]  # (bs, dim)

        pooled = F.relu(self.linear_layer(pooled))  # (batch_size,  embed_dim)
        #pooled = self.dropout(pooled)  # (batch_size, len_sentence, embed_dim)

        out_logits = self.classifier(pooled)  # (batch_size,  embed_dim)

        is_impossible = input_enc.pop('is_impossible', None)
        if is_impossible is not None:
            is_impossible = torch.LongTensor(is_impossible)

        return out_logits, is_impossible

    def save(self, path: str, epoch: int, train_iter: float, optimizer, train_loss: float, eval_loss: float):
        """
        Persist model on disk.
        """

        logger.info(f"Saving checkpoint model to {path}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'eval_loss': eval_loss,
            'train_iter': train_iter
        }, path)

    def compute_loss(self, out_logits, is_impossible):
        """
        Loss function for question-answering task

        Parameters
        ----------
        out_logits (tensor) - logit corresponding to probability if answer is contained in the context (batch_size, 2)
        is_impossible (tensor) - true labels (batch_size, 1)

        Returns
        -------
        1D Tensor loss
        """


        loss_fun = nn.CrossEntropyLoss()

        out_logits_g = out_logits.to(self.device, non_blocking=True)
        is_impossible_g = is_impossible.to(self.device, non_blocking=True)

        loss = loss_fun(out_logits_g, is_impossible_g)

        return loss


if __name__ == '__main__':
    from transformers import DistilBertModel
    from preprocess import SquadClassifierPreprocessor

    sp = SquadClassifierPreprocessor()
    train_enc, val_enc = sp.get_encodings(random_sample_train=0.0005, random_sample_val=0.1, return_tensors="pt")

    # Decoding
    #    print(sp.tokenizer.decode(train_enc['input_ids'][0]))

    dbm = DistilBertModel.from_pretrained('distilbert-base-uncased', return_dict=True)
    model = QAClassifierModel(transformer_model=dbm, device=torch.device("cpu"))

    out_logits, is_impossible = model(train_enc)

    print(model.compute_loss(out_logits, is_impossible))

    print("End")
