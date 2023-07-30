import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt


pipe_lr = joblib.load(open('model.pkl','rb'))


def predict_emotions(doc):
    results = pipe_lr.predict([doc])
    return results[0]

def get_prediction_proba(doc):
    result = pipe_lr.predict_proba([doc])
    return result[0]

emoji_dict = {
    'anger':'😠','disgust':'😑','fear':'😥','surprise':'🤩',
    'joy':'😃','shame':'😓','neutral':'🙃','sadness':'😓😔'
}

def main():
    st.title('Emotion Classification App')

    menu = ['Home','Monitor','About']

    choice =st.sidebar.selectbox('Menu',menu)
    if(choice=='Home'):
        st.subheader('Home Emotion in Text')
        with st.form(key='Emotion classifier'):
            raw_text = st.text_area('Type Here')
            submit=st.form_submit_button("Predict")

        if submit:
            col1,col2 = st.columns(2)
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            with col1:
                st.success('Original Text')
                st.write(raw_text)
                st.success('Prediction')
                emoji_icon = emoji_dict[prediction]
                st.write(f"{prediction}:{emoji_icon}")


            with col2:
                st.success('Prediction Probability')
                # st.write(probability)
                proba_values = probability.tolist()  # Convert the NumPy array to a Python list
                proba_df_clean = pd.DataFrame({'emotions': pipe_lr.classes_, 'probability': proba_values})
                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability')
                st.altair_chart(fig, use_container_width=True)


    elif choice == 'Monitor':
        st.subheader('Monitor app')
    
    else:
        st.subheader('About')

if __name__ == "__main__":
    main()