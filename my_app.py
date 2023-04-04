import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img


st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('Photofilter for Ecosia')
st.text('Upload Images')

@st.cache(allow_output_mutation=True)
def load_model():
    model = ResNet50(weights='imagenet')
    return model

model = load_model()

def predict(image):
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    preds = model.predict(image)
    pred_class = tf.keras.applications.resnet50.decode_predictions(preds, top=1)[0][0][1]
    return pred_class

uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    results = []
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file)
        st.image(img, caption=f'Uploaded Image: {uploaded_file.name}')
        
        if st.button(f'PREDICT {uploaded_file.name}'):
            st.write('Result...')
            pred_class = predict(img)
            st.title(f'PREDICTED OUTPUT:{pred_class}')
            result = pred_class
            results.append({'image': uploaded_file.name, 'result': result})
    
    if results:
        df = pd.DataFrame(results)
        st.dataframe(df)
        st.download_button(label='Download Results', data=df.to_excel, file_name='myresults.xlsx', mime='application/vnd.ms-excel')

