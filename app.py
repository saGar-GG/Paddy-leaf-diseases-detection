# app.py
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

model_path = "../modelsss/efficientnetb3-paddy-disease-detection-97.2.h5" 
model = tf.keras.models.load_model(model_path)

class_labels = {
    0: 'bacterial_leaf_blight',
    1: 'bacterial_leaf_streak',
    2: 'bacterial_panicle_blight',
    3: 'blast',
    4: 'brown_spot',
    5: 'dead_heart',
    6: 'downy_mildew',
    7: 'hispa',
    8: 'normal',
    9: 'tungro'
}

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_disease(img_array):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    return class_labels[predicted_class]

def get_disease_info(prediction):
    disease_info = {
        'bacterial_leaf_blight': {
            'cause': 'Caused by the bacterium Xanthomonas oryzae pv. oryzae.',
            'major_reason': 'Favorable conditions like high humidity and warm temperatures.',
            'prevention': 'Use resistant varieties, practice crop rotation, and field sanitation.',
            'pesticides': 'Copper-based fungicides.',
            'prevention_meds': 'Streptomycin and copper compounds.',
            'stage': 'Visible symptoms occur at the early to mid-stage.'
        },
        'bacterial_leaf_streak': {
            'cause': 'Caused by the bacterium Xanthomonas oryzae pv. oryzicola.',
            'major_reason': 'Favorable conditions like warm and humid weather.',
            'prevention': 'Use resistant varieties, crop rotation, and seed treatment.',
            'pesticides': 'Copper-based bactericides.',
            'prevention_meds': 'Streptomycin and copper compounds.',
            'stage': 'Visible symptoms occur at the early to mid-stage.'
        },
        'bacterial_panicle_blight': {
            'cause': 'Caused by the bacterium Burkholderia glumae.',
            'major_reason': 'Warm and humid weather conditions.',
            'prevention': 'Use disease-free seeds and adopt proper water management.',
            'pesticides': 'None specific.',
            'prevention_meds': 'Seed treatment with antibiotics.',
            'stage': 'Visible symptoms occur during panicle development.'
        },
        'blast': {
            'cause': 'Caused by the fungus Magnaporthe oryzae.',
            'major_reason': 'Warm and humid weather with continuous rain.',
            'prevention': 'Use resistant varieties, balanced fertilization, and fungicides.',
            'pesticides': 'Triazole and strobilurin-based fungicides.',
            'prevention_meds': 'Carbendazim, Tricyclazole.',
            'stage': 'Visible symptoms occur during all stages.'
        },
        'brown_spot': {
            'cause': 'Caused by the fungus Cochliobolus miyabeanus.',
            'major_reason': 'Warm and humid weather with heavy dew formation.',
            'prevention': 'Use resistant varieties and balanced nitrogen fertilization.',
            'pesticides': 'Fungicides containing azoxystrobin, difenoconazole.',
            'prevention_meds': 'Seed treatment with fungicides.',
            'stage': 'Visible symptoms occur during mid to late stages.'
        },
        'dead_heart': {
            'cause': 'Caused by various factors like insect damage or fungal infection.',
            'major_reason': 'Unfavorable weather conditions, insect infestation.',
            'prevention': 'Implement pest control measures and monitor crop health regularly.',
            'pesticides': 'Depends on the specific cause; insecticides or fungicides.',
            'prevention_meds': 'Depends on the specific cause; insecticides or fungicides.',
            'stage': 'Visible symptoms occur during early stages.'
        },
        'downy_mildew': {
            'cause': 'Caused by the fungus Sclerospora oryzae.',
            'major_reason': 'Cool and humid weather conditions.',
            'prevention': 'Use resistant varieties and fungicides.',
            'pesticides': 'Fungicides containing metalaxyl, mancozeb.',
            'prevention_meds': 'Metalaxyl, Mancozeb.',
            'stage': 'Visible symptoms occur during all stages.'
        },
        'hispa': {
            'cause': 'Caused by the insect Dicladispa armigera.',
            'major_reason': 'Insect infestation, particularly in the early stages of crop growth.',
            'prevention': 'Use insecticides, crop rotation, and resistant varieties.',
            'pesticides': 'Insecticides containing chlorpyrifos, thiamethoxam.',
            'prevention_meds': 'Neem-based formulations.',
            'stage': 'Visible symptoms occur during early stages.'
        },
        'normal': {
            'message': 'Your crops are healthy.'
        },
        'tungro': {
            'cause': 'Caused by a complex of two viruses: Rice tungro bacilliform virus (RTBV) and Rice tungro spherical virus (RTSV).',
            'major_reason': 'Vector (green leafhoppers) infestation and virus transmission.',
            'prevention': 'Use resistant varieties and control vector population.',
            'pesticides': 'None specific for viral infections.',
            'prevention_meds': 'None specific for viral infections.',
            'stage': 'Visible symptoms occur during all stages.'
        }
    }

    return disease_info.get(prediction, {})

#  Now lets make streamlit app
st.title("Paddy Leaf Disease Detection")

uploaded_file = st.file_uploader("Choose a Paddy leaf image...", type="jpg")

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    img = Image.open(uploaded_file)
    img_array = preprocess_image(img)
    prediction = predict_disease(img_array)

    st.write(f"Prediction: {prediction}")

    disease_info = get_disease_info(prediction)

    if 'message' in disease_info:
        st.write(disease_info['message'])
    else:
        st.write("Disease Information:")
        st.write(f"**Cause:** {disease_info['cause']}")
        st.write(f"**Major reason:** {disease_info['major_reason']}")
        st.write(f"**Prevention:** {disease_info['prevention']}")
        st.write(f"**Pesticides:** {disease_info['pesticides']}")
        st.write(f"**Prevention medicines:** {disease_info['prevention_meds']}")
        st.write(f"**Stage:** {disease_info['stage']}")
