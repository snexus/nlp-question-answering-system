"""
Quick and dirty evaluation of answers matching the dev dataset.

This evaluation script is not perfect - it misses many correct answers, since better post-processing is required.
"""

import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

from inference import QAModelInference
from preprocess import SquadPlausibleAnswersPreprocessor


def evaluate_possible(correct_ans, predicted_ans):
    ans = predicted_ans['answer'].replace(" ##", "").replace("  ,", ",").replace(" ,", ",")

    if correct_ans['text'].lower() == ans:
        return 1
    return 0


inf = QAModelInference(models_path="model_checkpoint", plausible_model_fn="model_plausible.pt",
                       possible_model_fn="model_possible_only.pt")

sp = SquadPlausibleAnswersPreprocessor()

contexts, questions, answers, is_impossible = sp._read_squad(
    os.path.join("squad", 'dev-v2.0.json'),
    frac=1.0,
    include_impossible=True)

predicted_impossible, is_correct = [], []
n = 300

for context, question,answer in tqdm(zip(contexts[:n], questions[:n], answers[:n])):
    ans = inf.extract_answer(context, question)
    if ans['plausible_answer'] :
        predicted_impossible.append(1)
    else:
        proba_po = (np.max(ans['start_word_proba_possible_model'])+np.max(ans['end_word_proba_possible_model']))/2
        #proba_pl = (np.max(ans['start_word_proba_plausible_model'])+np.max(ans['end_word_proba_plausible_model']))/2
        proba = 1-proba_po


        predicted_impossible.append(proba)
    is_correct.append(evaluate_possible(answer, ans))


print(predicted_impossible)
print(is_impossible[:n])

print("How well model detects impossible questions: ")
print(roc_auc_score(is_impossible[:n], predicted_impossible))
print(average_precision_score(is_impossible[:n], predicted_impossible))

is_correct = np.array(is_correct)

select = [not bool(x) for x in is_impossible[:n]]

print("Accuracy overall: ", np.mean(is_correct))
print("Accuracy (only possible): ", np.mean(is_correct[select]))

