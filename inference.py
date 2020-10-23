import logging
import os

import torch
from transformers import DistilBertModel, DistilBertTokenizerFast

from model import QAModel

logging.basicConfig(level=logging.INFO)


class QAModelInference:
    """
    This class combines output of models trained on possible only question and possible+impossible questions
    to produce textual output and corresponding probabilities.
    """

    def __init__(self, models_path, plausible_model_fn, possible_model_fn, device="cpu"):
        self.plausible_model_fn = plausible_model_fn
        self.possible_model_fn = possible_model_fn
        self.models_path = models_path
        self.device = device

        models = self._check_load_models()
        self.possible_model = models[0]
        self.plausible_model = models[1]

        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    def _check_load_models(self) -> list:
        """
        Checks if models exist and loads into memory.

        Returns
        -------
        list [possible_model, plausble_model], containing initialized instances of PyTorch models.
        """

        models = []
        for model_name in (self.possible_model_fn, self.plausible_model_fn):
            fn = os.path.join(self.models_path, model_name)
            logging.info(f"Loading {fn}")
            if not os.path.exists(fn):
                raise FileExistsError(f"Model {fn} doesn't exist. Please run training first.")
            models.append(self.load_model(fn))
        return models

    def load_model(self, state_path):
        """
        Initialises the model and loads saved state into the instance of the model.

        Parameters
        ----------
        state_path (str) - path pointing to the saved state.

        Returns
        -------
        Model (torch.nn.Module)
        """

        logging.info(f"Loading trained state from {state_path}")
        dbm = DistilBertModel.from_pretrained('distilbert-base-uncased', return_dict=True)
        device = torch.device(self.device)
        dbm.to(device)
        model = QAModel(transformer_model=dbm, device=device)

        # checkpoint = torch.load(state_path, map_location=device)
        model.load_state_dict(torch.load(state_path))
        model.eval()  # Switch to evaluation mode

        return model

    def get_model_data(self, model, context: str, question: str):
        """
        Extracts start and stop locations, words and location probabilities given the model instance, tokenizer,
        context and question.

        Parameters
        ----------
        model - instance of either "possible" or "plausible" models
        context (str) - text containing the context.
        question (str) - text containing the question.

        Returns
        -------
        start_idx (int) - start index of answer, in the context.
        end_idx (int) - end index of answer, in the context.
        words (list) - list of strings, containing the answer
        start_probabilities (np.array) - probabilities over the starting positions of the answer.
        end_probabilities (np.array) - probabilities over the end positions of the answer
        """

        tokens = self.tokenizer(context, question, truncation=True, padding=True, return_tensors="pt")
        start_logit, _, end_logit, _ = model(tokens)

        # Convert to proper probabilities
        start_logit, end_logit = torch.softmax(start_logit, dim=1), torch.softmax(end_logit, dim=1)
        start_idx, end_idx = torch.argmax(start_logit), torch.argmax(end_logit) + 1

        words = ""
        if end_idx < start_idx:
            end_idx = torch.argmax(end_logit[0][start_idx:]) + 1
            logging.warning(f"Error: start_idx = {start_idx}, end_idx = {end_idx}")
        else:
            input_ids = tokens['input_ids'].squeeze(0)
            words = self.tokenizer.decode(token_ids=input_ids[start_idx:end_idx].to('cpu').numpy())

        return start_idx, end_idx, words, start_logit.detach().to('cpu').numpy(), end_logit.detach().to('cpu').numpy()

    def extract_answer(self, context: str, question: str):

        # Get data for possible answers
        start_po, end_po, words_po, start_proba_po, end_proba_po = self.get_model_data(self.possible_model, context,
                                                                                       question)

        start_pl, end_pl, words_pl, start_proba_pl, end_proba_pl = self.get_model_data(self.plausible_model, context,
                                                                                       question)

        if start_po != start_pl and end_po != end_pl:
            ans = self._form_answer("<ANSWER UNKNOWN>", '', start_proba_po, end_proba_po, start_proba_pl,
                                    end_proba_pl,
                                    start_po, end_po, start_pl, end_pl)

            # As a plausible answer, return one with highest probability
            if max(start_proba_po[0]) + max(end_proba_po[0]) > max(start_proba_pl[0]) + max(end_proba_pl[0]):
                ans['plausible_answer'] = words_po
            else:
                ans['plausible_answer'] = words_pl
            return ans

        return self._form_answer(words_po, '', start_proba_po, end_proba_po, start_proba_pl, end_proba_pl,
                                 start_po, end_po,
                                 start_pl, end_pl)

    def _form_answer(self, answer_possible, answer_plausible, start_proba_po, end_proba_po, start_proba_pl, end_proba_pl,
                     start_po, end_po, start_pl, end_pl):
        """
        Forms an output dictionary.
        """

        return {
            'answer': answer_possible,
            'plausible_answer': answer_plausible,
            'start_word_proba_possible_model': start_proba_po,
            'end_word_proba_possible_model': end_proba_po,
            'start_word_proba_plausible_model': start_proba_pl,
            'end_word_proba_plausible_model': end_proba_pl,
            'start_position_possible_model': start_po,
            'end_position_possible_model': end_po,
            'start_position_plausible_model': start_pl,
            'end_position_plausible_model': end_pl,

        }


def save_lean_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(state_path, device="cpu"):
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
    pass

    # model_possible = load_model("model_checkpoint/plausible_model_30000.pt")
    #
    # save_lean_model(model_possible, "model_checkpoint/model_plausible.pt")
    # inf = QAModelInference(models_path="model_checkpoint", plausible_model_fn="model_plausible.pt",
    #                        possible_model_fn="model_possible_only.pt")

    # model_ = load_model("model_checkpoint/model_possible_only.pt")
    #
    # model_p = load_model("model_checkpoint/model_plausible.pt")
    #
    # #
    # tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # context = "The Norman dynasty had a major political, cultural and military impact on medieval Europe and even the Near East. The Normans were famed for their martial spirit and eventually for their Christian piety, becoming exponents of the Catholic orthodoxy into which they assimilated. They adopted the Gallo-Romance language of the Frankish land they settled, their dialect becoming known as Norman, Normaund or Norman French, an important literary language. The Duchy of Normandy, which they formed by treaty with the French crown, was a great fief of medieval France, and under Richard I of Normandy was forged into a cohesive and formidable principality in feudal tenure. The Normans are noted both for their culture, such as their unique Romanesque architecture and musical traditions, and for their significant military accomplishments and innovations. Norman adventurers founded the Kingdom of Sicily under Roger II after conquering southern Italy on the Saracens and Byzantines, and an expedition on behalf of their duke, William the Conqueror, led to the Norman conquest of England at the Battle of Hastings in 1066. Norman cultural and military influence spread from these new European centres to the Crusader states of the Near East, where their prince Bohemond I founded the Principality of Antioch in the Levant, to Scotland and Wales in Great Britain, to Ireland, and to the coasts of north Africa and the Canary Islands."

    # question =  "Who ruled the duchy of Normandy?"
    # question = "What type of major impact did the Norman dynasty have on modern Europe?"

    # question = "Who was famed for their Christian spirit?"

    # question = "Who assimilted the Roman language?"
    # question = "What principality did William the conquerer found?"
    # question = "Who ruled the country of Normandy?"
    #
    # context = """In the course of the 10th century, the initially destructive incursions of Norse war bands into the rivers of France evolved into more permanent encampments that included local women and personal property. The Duchy of Normandy, which began in 911 as a fiefdom, was established by the treaty of Saint-Clair-sur-Epte between King Charles III of West Francia and the famed Viking ruler Rollo, and was situated in the former Frankish kingdom of Neustria. The treaty offered Rollo and his men the French lands between the river Epte and the Atlantic coast in exchange for their protection against further Viking incursions. The area corresponded to the northern part of present-day Upper Normandy down to the river Seine, but the Duchy would eventually extend west beyond the Seine. The territory was roughly equivalent to the old province of Rouen, and reproduced the Roman administrative structure of Gallia Lugdunensis II (part of the former Gallia Lugdunensis)."""
    #
    # question = "What did the French promises to protect Rollo and his men from?"
    # question = "What treaty was established in the 9th century?"
    # question = "When was the Duchy of Normandy founded?"
    # question = "Who established a treaty with King Charles the third of France?"

    # question = "What river originally bounded the Duchy"
    # question = "when did Nors encampments ivolve into destructive incursions?"
    #
    # context = """The Normans thereafter adopted the growing feudal doctrines of the rest of France, and worked them into a functional hierarchical system in both Normandy and in England. The new Norman rulers were culturally and ethnically distinct from the old French aristocracy, most of whom traced their lineage to Franks of the Carolingian dynasty. Most Norman knights remained poor and land-hungry, and by 1066 Normandy had been exporting fighting horsemen for more than a generation. Many Normans of Italy, France and England eventually served as avid Crusaders under the Italo-Norman prince Bohemund I and the Anglo-Norman king Richard the Lion-Heart."""
    # # question = "What was one of the Norman's major exports?"
    # # question = "What was one of the Norman's major imports?"
    # question = "Who's arristocracy eventually served as avid Crusaders?"
    #
    # context = """During what campaign did the Vargian and Lombard fight?", "id": "5ad3dbc6604f3c001a3ff3ec", "answers": [], "is_impossible": true}], "context": "Soon after the Normans began to enter Italy, they entered the Byzantine Empire and then Armenia, fighting against the Pechenegs, the Bulgars, and especially the Seljuk Turks. Norman mercenaries were first encouraged to come to the south by the Lombards to act against the Byzantines, but they soon fought in Byzantine service in Sicily. They were prominent alongside Varangian and Lombard contingents in the Sicilian campaign of George Maniaces in 1038\u201340. There is debate whether the Normans in Greek service actually were from Norman Italy, and it now seems likely only a few came from there. It is also unknown how many of the \"Franks\", as the Byzantines called them, were Normans and not other Frenchmen."""
    # question = """Who was the Normans' main enemy in Italy, the Byzantine Empire and Armenia?"""
    # # question = """Who entered Italy soon after the Byzantine Empire?"""
    #
    # #     context = """Frozen II, also known as Frozen 2, is a 2019 American 3D computer-animated musical fantasy film produced by Walt Disney Animation Studios. The 58th animated film produced by the studio, and the sequel to the 2013 film Frozen, it features the return of directors Chris Buck and Jennifer Lee, producer Peter Del Vecho, songwriters Kristen Anderson-Lopez and Robert Lopez, and composer Christophe Beck. Lee also returns as screenwriter, penning the screenplay from a story by her, Buck, Marc E. Smith, Anderson-Lopez, and Lopez,[2] while Byron Howard executive-produced the film.[a][1] Kristen Bell, Idina Menzel, Josh Gad, Jonathan Groff, and Ciarán Hinds reprised their roles, while they are joined by newcomers Sterling K. Brown, Evan Rachel Wood, Alfred Molina, Martha Plimpton, Jason Ritter, Rachel Matthews, and Jeremy Sisto.
    # #
    # # Set three years after the events of the first film,[9] the story follows Elsa, Anna, Kristoff, Olaf, and Sven, who embark on a journey beyond their kingdom of Arendelle in order to discover the origin of Elsa's magical powers and save their kingdom after a mysterious voice calls out to Elsa.[10][11][12][13][14]
    # #
    # # Frozen II's world premiere was held at the Dolby Theatre in Hollywood on November 7, 2019, followed by the film's release by Walt Disney Studios Motion Pictures in the United States on November 22, 2019. The film had the highest all-time worldwide opening for an animated film and went on to gross $1.45 billion worldwide, making it the third highest-grossing film of 2019, the 10th highest-grossing film of all time and the second highest-grossing animated film of all time, behind the remake of The Lion King, which was released the same year.[15] The film received generally positive reviews from critics. It won two Annie Awards for Outstanding Achievement for Animated Effects in an Animated Production and Outstanding Achievement for Voice Acting in an Animated Feature Production and a Visual Effects Society Award for Outstanding Effects Simulations in an Animated Feature. At the 92nd Academy Awards, the film received a nomination for Best Original Song for "Into the Unknown"."""
    # #
    # #     question = """how much money did frozen 2 make?"""
    # #     question = "who is the story about?"
    #
    # # context = "The English name \"Normans\" comes from the French words Normans/Normanz, plural of Normant, modern French normand, which is itself borrowed from Old Low Franconian Nortmann \"Northman\" or directly from Old Norse Nor\u00f0ma\u00f0r, Latinized variously as Nortmannus, Normannus, or Nordmannus (recorded in Medieval Latin, 9th century) to mean \"Norseman, Viking\"."
    # # question = "When was the Latin version of the word Norman first recorded?"
    #
    # context = """
    #     Frozen II, also known as Frozen 2, is a 2019 American 3D computer-animated musical fantasy film produced by Walt Disney Animation Studios. The 58th animated film produced by the studio, and the sequel to the 2013 film Frozen, it features the return of directors Chris Buck and Jennifer Lee, producer Peter Del Vecho, songwriters Kristen Anderson-Lopez and Robert Lopez, and composer Christophe Beck. Lee also returns as screenwriter, penning the screenplay from a story by her, Buck, Marc E. Smith, Anderson-Lopez, and Lopez,[2] while Byron Howard executive-produced the film.[a][1] Kristen Bell, Idina Menzel, Josh Gad, Jonathan Groff, and Ciarán Hinds reprised their roles, while they are joined by newcomers Sterling K. Brown, Evan Rachel Wood, Alfred Molina, Martha Plimpton, Jason Ritter, Rachel Matthews, and Jeremy Sisto.
    # Set three years after the events of the first film,[9] the story follows Elsa, Anna, Kristoff, Olaf, and Sven, who embark on a journey beyond their kingdom of Arendelle in order to discover the origin of Elsa's magical powers and save their kingdom after a mysterious voice calls out to Elsa.
    #     """
    #
    # question = "who directed frozen 2?"
    #
    # # tokens = tokenizer(context, question, truncation=True, padding=True, return_tensors="pt")
    # # start_logit, _, end_logit, _ = model_(tokens)
    # #
    # # start_idx, end_idx = torch.argmax(start_logit), torch.argmax(end_logit)+1
    # #
    # # if end_idx < start_idx:
    # #     raise Exception(f"Error: start_idx = {start_idx}, end_idx = {end_idx}")
    # #
    # # input_ids = tokens['input_ids'].squeeze(0)
    # #
    # # words = tokenizer.convert_ids_to_tokens(input_ids[start_idx:end_idx])
    # #
    # # print(words)
    # # print("Done.")
    #
    # ans = inf.extract_answer(context, question)
    # print("Question: ", question)
    # print(f"answer: {ans['answer']}, plausible_answer: {ans['plausible_answer']}")

    # start_idx, end_idx, words, start_proba, end_proba = get_answer(model_, tokenizer, context, question)
    # print("Model containing only possible questions: ", start_idx, end_idx, words)
    #
    # start_idx, end_idx, words, start_proba_p, end_proba_p = get_answer(model_p, tokenizer, context, question)
    # print("Model containing impossible questions: ", start_idx, end_idx, words)
    #
    # s_p = start_proba[0].dot(start_proba_p[0])
    # e_p = end_proba[0].dot(end_proba_p[0])
    # print(e_p, s_p)
