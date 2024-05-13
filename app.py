import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODELS = {
    "CNN from Scratch": {
        "path": "models/cnn_scratch-40.h5",
        "size": (128, 128)
    },
    "DNN": {
        "path": "models/dnn.h5",
        "size": (224, 224)
    },
    "Transfer Learning (VGG19)": {
        "path": "models/cnn_vgg19-100L2.h5",
        "size": (224, 224)
    },
    "Transfer Learning (ResNet151)": {
        "path": "models/ResNet152_0.001_5.h5",
        "size": (224, 224)
    }
}

def predict(model_name, img):
    # Load the model
    print(f"Loading model: {model_name}")
    model_path = MODELS[model_name]["path"]
    model = load_model(model_path)

    # Load the image with size depending on the model
    print("Loading image")
    img_size = MODELS[model_name]["size"]
    img = image.load_img(img, target_size=img_size)
    print("Image loaded, resizing and normalizing the image")
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, *img_size, 3)

    # Predict
    print("Predicting the image class")
    prediction = model.predict(img_array)
    prediction = "Benign" if prediction < 0.5 else "Melanoma"
    print(f"Prediction: {prediction}")

    return prediction

def main():
    # Favicon
    st.set_page_config(page_title="Skin Cancer Detector", page_icon="🩺")

    # Title
    st.title("🩺 Skin Cancer Detector")

    # Subtitle
    st.markdown("Image classification web app to predict skin cancer.")

    # File uploader
    uploaded_file = st.file_uploader("Upload an image of a suspicious skin mole", type=["jpg", "jpeg", "png"])

    # Check if the file is uploaded
    if uploaded_file is not None:
        # Center-align the image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=False)

        # Select the model
        model_name = st.selectbox("Select a model", list(MODELS.keys()))

        # Predict button
        if st.button("🤖 Predict with AI", type="primary"):
            # Placeholder
            placeholder = st.empty()
            placeholder.write("⏳ Analyzing your image with Artificial Intelligence...")

            # Predict
            prediction = predict(model_name, uploaded_file)
            placeholder.empty()

            # Prediction
            if prediction == "Benign":
                st.write(f"✅ **{prediction}**")
                st.write("This skin mole does not appear to be cancerous. However, the AI model is not 100% accurate. Please consult a dermatologist.")
            else:
                st.write(f"⚠️ **{prediction}**")
                st.write("This skin mole appears to be cancerous. Please consult a dermatologist immediately.")
    
    # Footer
    st.markdown("---")
    st.markdown("👩🏻‍💻 Made by: [María Belén Salgado](https://github.com/MARIABELENSB), [Blanca Martínez](https://github.com/blancamartnez), [Elena Cabrera](https://github.com/elena-cabrera)")

if __name__ == "__main__":
    main()
