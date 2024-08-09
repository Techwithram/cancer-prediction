import streamlit as st
import pickle
import time
import pandas as pd

data = pd.read_csv(r"C:\Users\krish\Downloads\data.csv")
pkl_path = "C:\\Users\\krish\\OneDrive\\Desktop\\ok\\model.pkl"

with open(pkl_path, 'rb') as file:
    rf = pickle.load(file)
st.title("Breast Cancer Prediction")
st.header("Training data set")
st.write(data)
with st.expander("see visualisation"):
 st.scatter_chart(data,y='diagnosis')
 st.divider()
 st.bar_chart(data,y="diagnosis")
 st.divider()

with st.sidebar:
 st.header("User input")
 text_me =st.slider('Texture mean',0.00,50.00,25.24)
 smo_me =st.slider('smoothness mean',0.00000,1.00000)
 com_me =st.slider('compactness_mean',0.00000,1.00000)
 con_po_me =st.slider('concave points_mean',0.00000,1.00000)
 sym_me =st.slider('symmetry_mean',0.00000,1.00000)
 fra_me = st.slider('fractal_dimension_mean',0.00000,1.00000)
 text_se =st.slider('texture_se',0.00000,1.00000)
 are_se =st.slider('area_se',0.00000,1.00000)
 smo_se =st.slider('smoothness_se',0.00000,1.00000)
 com_se =st.slider('compactness_se',0.00000,1.00000)
 con_se =st.slider('concavity_se',0.00000,1.00000)
 con_po_se =st.slider('concave points_se',0.00000,1.00000)
 sym_se =st.slider('symmetry_se',0.00000,1.00000)
 fra_se =st.slider('fractal_dimension_se',0.00000,1.00000)
 text_wo = st.slider('texture_worst',0.00000,1.00000)
 are_wo =st.slider('area_worst',0.00000,1.00000)
 smo_wo =st.slider('smoothness_worst',0.00000,1.00000)
 com_wo =st.slider('compactness_worst',0.00000,1.00000)
 con_wo=st.slider('concavity_worst',0.00000,1.00000)
 con_po_wo =st.slider('concave points_worst',0.00000,1.00000)
 sym_wo =st.slider('symmetry_worst',0.00000,1.00000)
 fra_wo =st.slider('fractal_dimension_worst',0.00000,1.00000)

 user_input = {
  'texture_mean':text_me,
   'smoothness_mean':smo_me,
   'compactness_mean':com_me,
   'concave points_mean':con_po_me,
   'symmetry_mean':sym_me,
   'fractal_dimension_mean':fra_me,
   'texture_se':text_se,
    'area_se':are_se,
    'smoothness_se':smo_se, 'compactness_se':com_se,
       'concavity_se':con_se, 'concave points_se':con_po_se, 'symmetry_se':sym_se,
       'fractal_dimension_se':fra_se, 'texture_worst':text_wo, 'area_worst':are_wo,
       'smoothness_worst':smo_wo, 'compactness_worst':com_wo, 'concavity_worst':con_wo,
       'concave points_worst':con_po_wo, 'symmetry_worst':sym_wo, 'fractal_dimension_worst':fra_wo
 }
 user_report = pd.DataFrame(data=user_input,index=[0])
 on =st.button("predict report")
 if on:
  with st.spinner("your report is being processed"):
   time.sleep(7)
   st.subheader("Your report:")
   if rf.predict(user_report) == 0:
    st.write("Your tumor is benign")
   else:
    st.write("Your tumor is malignant")