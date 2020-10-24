import os
import urllib.request

import numpy as np
import pandas as pd
import streamlit as st

import config as cfg
from inference import QAModelInference


def get_proba(ans_dict: dict, model_tag: str):
    """
    Returns probability distribution over start and end words, together with confidence.

    Parameters
    ----------
    ans - dictionary containing inference result.
    model_tag - string denoting the model, can be either "possible" or "plausible"

    Returns
    -------
    start_praba - nd.array containing probability distribution over start word
    end_praba - nd.array containing probability distribution over end word
    p - float, combined probability of highest probable start/end words.
    """

    start_proba_ = ans_dict[f'start_word_proba_{model_tag}_model'][0]
    end_proba_ = ans_dict[f'end_word_proba_{model_tag}_model'][0]
    p = (np.max(start_proba_) + np.max(end_proba_)) / 2
    return start_proba_, end_proba_, p


def fetch_cache_models():
    """
    If models don't exits on dist, download and store them.
    This is due to Streamlit Sharing current limiations (Oct 2020)
    """

    folder = cfg.model_folder
    if not os.path.exists(folder):
        os.makedirs(folder)

    for model_name, url in cfg.mappings.items():
        fn = f"{model_name}.pt"
        if not os.path.exists(os.path.join(folder, fn)):
            urllib.request.urlretrieve(url, os.path.join(folder, fn))


@st.cache(allow_output_mutation=True)
def load_model():
    inf = QAModelInference(models_path=cfg.model_folder, plausible_model_fn="model_plausible.pt",
                           possible_model_fn="model_possible.pt")
    return inf


with st.spinner("Caching models..."):
    fetch_cache_models()

with st.spinner("Loading models..."):
    model = load_model()

st.title("Question Answering System")

st.write("Built with 🤗 Transformers")

with st.beta_expander("About"):
    st.markdown("""
    The main goal of the project is to learn working with 🤗 transformers architecture by replacing the default head with a custom head suitable for the task, and fine-tuning using custom data.
    In addition, the project tries to improve on the ability to recognise tricky (impossible) questions which are part of SQuAD 2.0 dataset. 
    This project doesn't use QA task head coming with HuggingFace transformers but creates the head architecture from scratch.
     The same architecture is used to fine-tune 2 models, as described below.

The QA system is built using several sub-components:
* HuggingFace's DistilBERT transformer with custom head, fine-tuned on SQuAD v2.0, using only possible questions.
* HuggingFace's DistilBERT transformer with custom head, fine-tuned on SQuAD v2.0, using both - possible and non-possible questions.
* Inference component, combining the output of both models.

The logic behind training two models - the former is a conditional model, trained only on correct question/answers pairs, while the latter additionally includes tricky questions with answers that can't be found in the context. The idea is that combining the output of both models will improve the discrimination ability on impossible questions.
    """)

st.info(
    ":bulb: How does it work? Enter a context, then ask a question about it."
    " The system will attempt to find and extract the answer, given it exists in the context.")

example_context = """In 1984, Time Magazine reported that Sassafras, a female poodle belonging to a New York City physician, had received a diploma from the American Association of Nutrition and Dietary Consultants.  Her owner had bought the diploma for $50 to demonstrate that "something that looks like a diploma doesn't mean that somebody has responsible training"."""

example_question = "Who Sassafras belongs to?"



st.subheader("Context")
context = st.text_area("Provide context", value=example_context, height = 150, max_chars=3000)
#st.markdown(context)

st.subheader("Question")
question = st.text_input("Ask a question", value=example_question)


#st.markdown(question)
st.subheader("Model")
model_selection = st.selectbox("Choose a model", options=['Automatic', 'Trained on correct questions',
                                                                  'Trained on tricky questions'])


if st.button("Get an answer"):

    ans = model.extract_answer(context, question)

    if model_selection == 'Automatic':
        s_p, e_p, pr_p = get_proba(ans, model_tag="possible")
        s_pl, e_pl, pr_pl = get_proba(ans, model_tag="plausible")
        #print(ans)
        if ans['plausible_answer'] != '':
            start_p, end_p, confidence, answer = s_pl, e_pl, pr_pl, f"Models didn't agee on the answer. Choosing most probable: " \
                                                                    f"\"{ans['plausible_answer']}\"."
        else:
            start_p, end_p, confidence, answer = s_p, e_p, pr_p, ans['answer']
    elif model_selection == 'Trained on correct questions':
        start_p, end_p, confidence = get_proba(ans, model_tag="possible")
        answer = ans['answer']
    else:
        start_p, end_p, confidence = get_proba(ans, model_tag="plausible")
        answer = ans['plausible_answer']

    st.subheader("Answer")
    if not answer:
        st.markdown("Can't determine the answer.")
    else:
        st.markdown(answer)

    st.markdown("**Confidence**: {:.3f}".format(confidence))

    # print(ans['start_word_proba_possible_model'][0])

    st.markdown("---")
    st.markdown("**Probability distributions of start/end indices**")
    df = pd.DataFrame(columns=['start', 'end'])
    df['start'] = start_p
    df['end'] = end_p
    st.bar_chart(df)
    # st.bar_chart(ans['start_word_proba_possible_model'][0])
    # st.bar_chart(ans['end_word_proba_possible_model'][0])
