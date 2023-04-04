import numpy as np
import zipfile
from skimage.io import imread
from skimage.transform import resize
from PIL import Image
import streamlit as st
import pickle
import pandas as pd

st.title('Photofilter for Ecosia')
st.text('Upload Images')

model=pickle.load(open('rs_rf.pkl','rb'))

uploaded_files = st.file_uploader("Choose images...", type="jpg", accept_multiple_files=True)

if uploaded_files:
    results = []
    for uploaded_file in uploaded_files:
        img=Image.open(uploaded_file)
        st.image(img, caption=f'Uploaded Image: {uploaded_file.name}')
        
        if st.button(f'PREDICT {uploaded_file.name}'):
            CATEGORIES=['bad','good']
            st.write('Result...')
            flat_data = []
            img = np.array(img)
            img_resized = resize(img,(150,150,3))
            flat_data.append(img_resized.flatten())
            flat_data = np.array(flat_data)
            y_out = model.predict(flat_data)
            y_out = CATEGORIES[y_out[0]]
            st.title(f'PREDICTED OUTPUT:{y_out}')
            result = y_out
            results.append({'image': uploaded_file.name, 'result': result})
    
    if results:
        df = pd.DataFrame(results)
        st.dataframe(df)
        st.download_button(label='Download Results', data=df.to_excel, file_name='myresults.xlsx', mime='application/vnd.ms-excel')