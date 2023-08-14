import numpy as np
import pickle
import streamlit as st
from classifier import Classifier

loaded_model = None
with open('best_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

def prediction(model, sepal_length,sepal_width,petal_length,petal_width):
    X_new = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    X_new = X_new.reshape(1,-1)
    prediction = model.predict(X_new)
    if prediction == 0:
        return 'This is Iris-setosa','https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
    elif prediction == 1:
        return 'This is Iris-versicolor','https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
    else:
        return 'This is Iris-virginica' , 'https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'

def main():
    st.title('Iris Prediction')
    st.title('20020541 - Quách Xuân Nam')
    sepal_length = st.text_input('Length of sepal')
    sepal_width = st.text_input('Width of sepal')
    petal_length = st.text_input('Length of petal')
    petal_width = st.text_input('Width of Petal')

    prediction_iris = ''
    flower_img = ''
    if st.button('Predict'):
        prediction_iris,flower_img = prediction(loaded_model, sepal_length,sepal_width,petal_length,petal_width)
        st.success(prediction_iris)
        st.image(flower_img, width=300)

if __name__ == '__main__':
    main()