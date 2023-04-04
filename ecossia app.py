import numpy as np
import zipfile
from skimage.io import imread
from skimage.transform import resize
from PIL import Image
import streamlit as st
st.title('Photofilter for Ecosia')
st.text('Upload the Image')
import pickle

model=pickle.load(open('rs_rf.pkl','rb'))

uploaded_file = st.file_uploader("Choose an image...",type="jpg",accept_multiple_files=False)
if uploaded_file is not None:
  img=Image.open(uploaded_file)
  st.image(img,caption='Uploaded Image')
  if st.button('PREDICT'):
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
    result=y_out
    st.download_button(label='Download Results',data=result,file_name='myresults.xlsx')
    
    

      




    
  





    


