import streamlit as st


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

        # Predict button
        if st.button("🤖 Predict with AI", type="primary"):
            # Display the prediction
            st.write("⏳ Analyzing your image with Artificial Intelligence...")

            # Prediction
            prediction = "Melanoma"
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
