# Question Answering System

*Work In Progress*

This repository contains an implementation of the question-answering system. The main goal of the project is to learn working
with 🤗 transformers architecture by replacing the default head with a custom head suitable for the task, and fine-tuning using custom data.
In addition, the project tries to improve on the ability to recognise tricky (impossible) questions which are part of SQuAD 2.0 dataset.
This project **doesn't use** QA task head coming with HuggingFace transformers but creates the head architecture from scratch.
The same architecture is used to fine-tune 2 models, as described below. 

The QA system is built using several sub-components:
* HuggingFace's DistilBERT transformer with custom head, fine-tuned on SQuAD v2.0, using only possible questions.
* HuggingFace's DistilBERT transformer with custom head, fine-tuned on SQuAD v2.0, using both - possible and non-possible questions.
* Inference component, combining the output of both models.

The logic behind training two models - the former is a conditional model, trained only on correct question/answers pairs, 
while the latter additionally includes tricky questions with answers that can't be found in the context. 
The idea is that combining the output of both models will improve the discrimination ability on impossible questions.

## Web application 
https://share.streamlit.io/snexus/nlp-question-answering-system/main

## Installation and running

*  Clone the repository.

*  Create and activate conda environment:
```shell script
conda env create -f environment.yml
conda activate nlp-question-answering-system
```

* Download the trained models:
TODO - store on publicly available service.

* To explore the predictions, use demo web app built using Streamlit.
```python
streamlit run ./webapp.py
```

### Training

Download the SQuAD 2.0 dataset for training

```
mkdir squad
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -O squad/train-v2.0.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O squad/dev-v2.0.json
```

#### Training locally

```python
import train
from model import QAModel

from preprocess import SquadPreprocessor, SquadPlausibleAnswersPreprocessor

train_model(preprocessor=SquadPreprocessor, base_model=QAModel, frac_val_data=0.025, frac_train_data=0.025, batch_size = 8, n_epoch = 3, 
                force_cpu=True)
```
For more information about training options, use ```help(train_model)```.

#### Training on Google Colab
GC allows using GPU accelerated training by using GPU enabled runtime. To change runtime type, use Runtime-> Change runtime type.

To train with GC, use `google_colab_train.ipynb` from the notebooks folder.