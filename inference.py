# TODO Refactor into proper class

import logging

import torch
from transformers import DistilBertModel, DistilBertTokenizerFast

from model import QAModel

logging.basicConfig(level=logging.INFO)


def load_model(state_path, device = "cpu"):
    logging.info(f"Loading trained state from {state_path}")
    dbm = DistilBertModel.from_pretrained('distilbert-base-uncased', return_dict=True)
    device = torch.device(device)
    dbm.to(device)
    model = QAModel(transformer_model=dbm, device=device)


    checkpoint = torch.load(state_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Switch to evaluation mode

    return model


if __name__ == '__main__':
    model_ = load_model("model_checkpoint/model_30000.pt")

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    context = "The Norman dynasty had a major political, cultural and military impact on medieval Europe and even the Near East. The Normans were famed for their martial spirit and eventually for their Christian piety, becoming exponents of the Catholic orthodoxy into which they assimilated. They adopted the Gallo-Romance language of the Frankish land they settled, their dialect becoming known as Norman, Normaund or Norman French, an important literary language. The Duchy of Normandy, which they formed by treaty with the French crown, was a great fief of medieval France, and under Richard I of Normandy was forged into a cohesive and formidable principality in feudal tenure. The Normans are noted both for their culture, such as their unique Romanesque architecture and musical traditions, and for their significant military accomplishments and innovations. Norman adventurers founded the Kingdom of Sicily under Roger II after conquering southern Italy on the Saracens and Byzantines, and an expedition on behalf of their duke, William the Conqueror, led to the Norman conquest of England at the Battle of Hastings in 1066. Norman cultural and military influence spread from these new European centres to the Crusader states of the Near East, where their prince Bohemond I founded the Principality of Antioch in the Levant, to Scotland and Wales in Great Britain, to Ireland, and to the coasts of north Africa and the Canary Islands."

    # question =  "Who ruled the duchy of Normandy?"
    #question = "What religion were the Normans?"
    #question = "Who was the duke in the battle of Hastings?"  # "William the Conqueror"
    #question = "Who was famed for their Christian spirit?"

    # context = "The English name \"Normans\" comes from the French words Normans/Normanz, plural of Normant, modern French normand, which is itself borrowed from Old Low Franconian Nortmann \"Northman\" or directly from Old Norse Nor\u00f0ma\u00f0r, Latinized variously as Nortmannus, Normannus, or Nordmannus (recorded in Medieval Latin, 9th century) to mean \"Norseman, Viking\"."
    # question = "When was the Latin version of the word Norman first recorded?"

#     context = """
#     Frozen II, also known as Frozen 2, is a 2019 American 3D computer-animated musical fantasy film produced by Walt Disney Animation Studios. The 58th animated film produced by the studio, and the sequel to the 2013 film Frozen, it features the return of directors Chris Buck and Jennifer Lee, producer Peter Del Vecho, songwriters Kristen Anderson-Lopez and Robert Lopez, and composer Christophe Beck. Lee also returns as screenwriter, penning the screenplay from a story by her, Buck, Marc E. Smith, Anderson-Lopez, and Lopez,[2] while Byron Howard executive-produced the film.[a][1] Kristen Bell, Idina Menzel, Josh Gad, Jonathan Groff, and Ciarán Hinds reprised their roles, while they are joined by newcomers Sterling K. Brown, Evan Rachel Wood, Alfred Molina, Martha Plimpton, Jason Ritter, Rachel Matthews, and Jeremy Sisto.
# Set three years after the events of the first film,[9] the story follows Elsa, Anna, Kristoff, Olaf, and Sven, who embark on a journey beyond their kingdom of Arendelle in order to discover the origin of Elsa's magical powers and save their kingdom after a mysterious voice calls out to Elsa.
#     """
#
#     question = "What is frozen 2?"

    tokens = tokenizer(context, question, truncation=True, padding=True, return_tensors="pt")
    start_logit, _, end_logit, _ = model_(tokens)

    start_idx, end_idx = torch.argmax(start_logit), torch.argmax(end_logit)+1

    if end_idx < start_idx:
        raise Exception(f"Error: start_idx = {start_idx}, end_idx = {end_idx}")

    input_ids = tokens['input_ids'].squeeze(0)

    words = tokenizer.convert_ids_to_tokens(input_ids[start_idx:end_idx])

    print(words)
    print("Done.")
