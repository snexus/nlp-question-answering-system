import logging

import torch
import torch.nn as nn

logger = logging.getLogger()


class QAModel(nn.Module):
    def __init__(self, transformer_model, device, dropout_proba=0.2):
        super().__init__()
        self.device = device  # if torch.cuda.is_available() else torch.device('cpu') #torch.device('cpu')
        self.transformer = transformer_model
        self.embed_dim = self.transformer.config.dim

        # We replace the head with linear layer
        self.qa_head = nn.Linear(in_features=self.embed_dim, out_features=2, bias=True)
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
        start_logit - logit corresponding to the start position of the answer (batch_size, sentence_size, 1)
        start_pos - true start position (batch_size, 1) or None
        end_logit - logit corresponding to the end position of the answer (batch_size, sentence_size, 1)
        end_pos - ture end position (batch_size, 1) or None
        """

        # For real data, start and end positions won't be present
        start_pos = input_enc.pop('start_positions', None)
        end_pos = input_enc.pop('end_positions', None)

        # For training, transform start and end position lists into tensors
        input_ids = input_enc['input_ids'].to(self.device)
        attention_mask = input_enc['attention_mask'].to(self.device)

        trans_out = self.transformer(input_ids, attention_mask=attention_mask)

        # Extract hidden state from the transformer
        hidden_out = trans_out.last_hidden_state  # (batch_size, len_sentence, embed_dim)
        hidden_out = self.dropout(hidden_out)  # (batch_size, len_sentence, embed_dim)

        # Pass through the linear layer, we need to learn it's parameters
        out = self.qa_head(hidden_out)  # (batch_size, len_sentence, 2)

        start_logit, end_logit = out.split(1, dim=-1)
        start_logit = start_logit.squeeze(-1)  # (bs, max_query_len)
        end_logit = end_logit.squeeze(-1)

        # # Pass through classification
        # out_cls = F.relu(self.class_layer(hidden_out)).permute(2, 0, 1)  # (1, batch_size, len_sentence)
        # out_cls_logits = F.max_pool1d(out_cls, kernel_size=out_cls.shape[-1]).squeeze(-1).permute(1,
        #                                                                                           0)  # (batch_size, 2)

        if start_pos is not None and end_pos is not None:
            start_pos = torch.LongTensor(start_pos)
            end_pos = torch.LongTensor(end_pos)
            # start_pos = torch.tensor(start_pos, dtype=torch.long).reshape(-1, 1).squeeze(-1)
            # end_pos = torch.tensor(end_pos, dtype=torch.long).reshape(-1, 1).squeeze(-1)  # (batch_size)

            # is_impossible = torch.tensor(is_impossible, dtype=torch.long).detach().reshape(-1, 1).squeeze(
            #     -1)  # (batch_size)

            # ignored_index = start_logit.size(1)
            # start_pos.clamp_(0, ignored_index)
            # end_pos.clamp_(0, ignored_index)

        return start_logit, start_pos, end_logit, end_pos  # , ignored_index

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

    def compute_loss(self, start_logit, start_pos, end_logit, end_pos):
        """
        Loss function for question-answering task

        Parameters
        ----------

        start_logit - logit corresponding to the start position of the answer (batch_size, sentence_size, 1)
        start_pos - true start position (batch_size, 1)
        end_logit - logit corresponding to the end position of the answer (batch_size, sentence_size, 1)
        end_pos - ture end position (batch_size, 1)

        Returns
        -------
        1D Tensor loss
        """
        # cls_loss = loss_fun(out_cls_logits, is_impossible)

        ignored_index = start_logit.size(1)
        start_pos.clamp_(0, ignored_index)
        end_pos.clamp_(0, ignored_index)

        loss_fun = nn.CrossEntropyLoss(ignore_index=ignored_index)

        start_logit_g = start_logit.to(self.device, non_blocking=True)
        end_logit_g = end_logit.to(self.device, non_blocking=True)
        start_pos_g = start_pos.to(self.device, non_blocking=True)
        end_pos_g = end_pos.to(self.device, non_blocking=True)

        start_loss = loss_fun(start_logit_g, start_pos_g)
        end_loss = loss_fun(end_logit_g, end_pos_g)

        return (start_loss + end_loss) / 2


if __name__ == '__main__':
    # from transformers import DistilBertModel
    # from preprocess import SquadPreprocessor
    #
    # sp = SquadPreprocessor()
    # train_enc, val_enc = sp.get_encodings(random_sample_train=0.0005, random_sample_val=0.1, return_tensors="pt")
    #
    # # Decoding
    # #    print(sp.tokenizer.decode(train_enc['input_ids'][0]))
    #
    # dbm = DistilBertModel.from_pretrained('distilbert-base-uncased', return_dict=True)
    # model = QAModel(transformer_model=dbm, device=torch.device("cpu"))
    #
    # start_logit, start_pos_, end_logit, end_pos_, out_cls_logits, is_impossible = model(train_enc)
    # loss_fun_ = nn.CrossEntropyLoss()
    # print(model.compute_loss(start_logit, start_pos_, end_logit, end_pos_, out_cls_logits))

    print("End")
