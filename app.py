import numpy as np
import pickle
import streamlit as st
from classifier import Classifier

st.set_page_config(
    page_title="Flower Classifier App",
    page_icon="🌻",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': '''Quách Xuân Nam - 20020541 - IUH\n
        https://www.facebook.com/20020541.nam'''
    }
)
model = None
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict(features):
    decode = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    X_new = np.array([features]).reshape(1,-1)
    prediction = model.predict(X_new)
    return decode[prediction[0]]
    
def input_features():
    sepal_length = st.sidebar.number_input('Sepal length', min_value=4.3, max_value=7.9, value=5.4, step=0.1)
    # sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    
    sepal_width = st.sidebar.number_input('Sepal width', min_value=2.0, max_value=4.4, value=3.4, step=0.1)
    # sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    
    petal_length = st.sidebar.number_input('Petal length', min_value=1.0, max_value=6.9, value=1.3, step=0.1)
    # petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    
    petal_width = st.sidebar.number_input('Petal width', min_value=0.1, max_value=2.5, value=0.2, step=0.1)
    # petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    return [sepal_length,sepal_width,petal_length,petal_width]

def main():
    st.image('images\\backgound.png', use_column_width='always', width=20)
    st.markdown('<h1 style="text-align: center;">🌷 Flower Classification</h1>', unsafe_allow_html=True)
    st.markdown('---')
    data = input_features()
    pred = predict(data)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<h1 style="text-align: center;">Image flower</h1>', unsafe_allow_html=True)
        st.image(f'images\\{pred}_flower.png', width=600)

    with col2:
        st.markdown('<h1 style="text-align: center;">Prediction</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="text-align: center; border: 5px solid green; ">{pred}</h1>', unsafe_allow_html=True)
    st.caption('Modify by :blue[qxnam]')
if __name__ == '__main__':
    main()