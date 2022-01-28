from pycaret.regression import load_model, predict_model
import streamlit as st
import numpy as np
import pandas as pd

# Cargar el modelo

model = load_model('FinalInsuranceModel26-01-2022')

def run():
    from PIL import Image
    image_integral = Image.open('is_icon1.png')

    st.title('Predicción Gastos en Salud')
    st.sidebar.info('Esta App ha sido creada con PyCaret y Streamlit')
    st.sidebar.image(image_integral)

    # Data Inputs

    age = st.number_input('Edad', min_value=1, max_value=120, value=21)
    sex = st.selectbox('Sexo', ('Masculino', 'Femenino'))
    bmi = st.number_input('IMC', min_value=10, max_value=50, value=20)
    children = st.selectbox('Hijos', (0,1,2,3,4,5,6,7,8,9,10))
    
    if st.checkbox('Fumador'):
        smoker = 'yes'
    else:
        smoker = 'no'

    region = st.selectbox('Región', ('southwest', 'northwest', 'southeast', 'northeast'))

    output = ''

    input_dictionary = {'age':age, 'sex':sex, 'bmi':bmi, 'children':children, 'smoker':smoker,
    'region':region}
    input_df = pd.DataFrame([input_dictionary])

    # Predicción
    if st.button('Predecir'):
        output = predict_model(model, data=input_df)
        output = '$' + str(output['Label'][0])

    # Mostrar la predicción
    st.success('La predicción sobre el gasto es {}'.format(output))

if __name__ == '__main__':
    run()

