# Question Answering System

*Work In Progress*

This repository contains an implementation of the question-answering system. The main goal of the project is to learn working
with 🤗 Transformers architecture by replacing the default head with a custom head suitable for the task, and fine-tuning using custom data.

The QA system is built using several sub-components:
* HuggingFace's DistilBERT transformer with custom head, fine-tuned on SQuAD v2.0, using only possible questions.
* HuggingFace's DistilBERT transformer with custom head, fine-tuned on SQuAD v2.0, using both - possible and non-possible questions.
* Inference component, combining the output of both models.

## Installation and running

### Download SQuAD 2.0 data
```
mkdir squad
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -O squad/train-v2.0.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O squad/dev-v2.0.json
```

### Training

Example training the QA model for possible questions:

```python
import train
from model import QAModel

from preprocess import SquadPreprocessor, SquadPlausibleAnswersPreprocessor

train_model(preprocessor=SquadPreprocessor, base_model=QAModel, frac_val_data=0.025, frac_train_data=0.025, batch_size = 8, n_epoch = 3, 
                force_cpu=True)
```
For more information about training options, use ```help(train_model)```.

### Training on Google Colab
GC allows using GPU accelerated training by using GPU enabled runtime. To change runtime type, use Runtime-> Change runtime type.

To train the models, use `google_colab_train.ipynb`.