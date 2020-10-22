import streamlit as st
import urllib.request
from inference import QAModelInference

import os
import config as cfg


def fetch_cache_models():
    folder = cfg.model_folder
    if not os.path.exists(folder):
        os.makedirs(folder)

    for model_name, url in cfg.mappings.items():
        fn = f"{model_name}.pt"
        if not os.path.exists(os.path.join(folder, fn)):
            urllib.request.urlretrieve(cfg.possible_model_url, os.path.join(folder, fn))


@st.cache(allow_output_mutation=True)
def load_model():
    inf = QAModelInference(models_path="models", plausible_model_fn="model_plausible.pt",
                           possible_model_fn="model_possible.pt")
    return inf


with st.spinner("Caching models..."):
    fetch_cache_models()



model = load_model()
st.title("Question Answering System")
st.markdown("---")

st.info(
    "How it works? Enter a context, then ask a question about it."
    " The system should extract a text paragraph containing an answer.")

context = st.text_area("Enter context", max_chars=3000)
st.markdown(context)
question = st.text_input("Enter question")

if st.button("Get an answer"):
    st.markdown("---")
    ans = model.extract_answer(context, question)
    st.markdown("**Answer:** " + ans['answer'])
    if ans['plausible_answer']:
        st.markdown("**Plausible Answer:** " + ans['plausible_answer'])

    #print(ans['start_word_proba_possible_model'][0])

    st.subheader("Output Probability Distributions")
    st.bar_chart(ans['start_word_proba_possible_model'][0])
    st.bar_chart(ans['end_word_proba_possible_model'][0])
