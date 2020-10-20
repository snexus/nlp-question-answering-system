import streamlit as st

from inference import QAModelInference


@st.cache(allow_output_mutation=True)
def load_model():
    inf = QAModelInference(models_path="model_checkpoint", plausible_model_fn="model_plausible.pt",
                           possible_model_fn="model_possible_only.pt")
    return inf


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
