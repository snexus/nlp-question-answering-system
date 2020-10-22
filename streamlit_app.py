import streamlit as st
import urllib.request
from inference import QAModelInference

import os
import config as cfg
import pandas as pd
import numpy as np


def fetch_cache_models():
    """
    If models don't exits on dist, download and store them.
    """

    folder = cfg.model_folder
    if not os.path.exists(folder):
        os.makedirs(folder)

    for model_name, url in cfg.mappings.items():
        fn = f"{model_name}.pt"
        if not os.path.exists(os.path.join(folder, fn)):
            urllib.request.urlretrieve(cfg.possible_model_url, os.path.join(folder, fn))


@st.cache(allow_output_mutation=True)
def load_model():
    inf = QAModelInference(models_path=cfg.model_folder, plausible_model_fn="model_plausible.pt",
                           possible_model_fn="model_possible.pt")
    return inf


with st.spinner("Caching models..."):
    fetch_cache_models()



model = load_model()
st.title("Question Answering System")

st.info(
    ":bulb: How does it work? Enter a context in the control panel, then ask a question about it."
    " The system will attempt to find and extract the answer.")

example_context = """In 1984, Time Magazine reported that Sassafras, a female poodle belonging to a New York City
 physician, had received a diploma from the American Association of Nutrition and Dietary Consultants. 
 Her owner had bought the diploma for $50 to demonstrate that "something that looks like a diploma doesn't mean
  that somebody has responsible training"."""

example_question = "Where did Sassafras live?"

st.sidebar.subheader("Control Panel")
context = st.sidebar.text_area("Provide a context", value=example_context, max_chars=3000)

st.subheader("Context")
st.markdown(context)


question = st.sidebar.text_input("Enter a question", value=example_question)

st.subheader("Question")
st.markdown(question)

if st.sidebar.button("Get an answer"):

    ans = model.extract_answer(context, question)


    st.subheader("Answer")
    if not ans['answer']:
        st.markdown("Can't determine the answer.")
    else:
        st.markdown(ans['answer'])

    if ans['plausible_answer']:
        st.subheader("Plausible Answer")
        st.markdown(ans['plausible_answer'])

    confidence = (np.max(ans['start_word_proba_possible_model']) + np.max(ans['end_word_proba_possible_model']))/2
    st.markdown("**Confidence**: {:.3f}".format(confidence))

    #print(ans['start_word_proba_possible_model'][0])

    st.markdown("---")
    st.markdown("**Probability distributions of start/end indices**")
    df = pd.DataFrame(columns = ['start', 'end'])
    df['start'] = ans['start_word_proba_possible_model'][0]
    df['end'] = ans['end_word_proba_possible_model'][0]
    st.bar_chart(df)
    # st.bar_chart(ans['start_word_proba_possible_model'][0])
    # st.bar_chart(ans['end_word_proba_possible_model'][0])

