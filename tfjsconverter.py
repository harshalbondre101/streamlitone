import streamlit as st
import tensorflow as tf
import tensorflowjs as tfjs

st.title("TensorFlow Saved Model to TensorFlow.js Converter")

# Sidebar for model input
st.sidebar.header("Choose Your Model")

uploaded_model = st.sidebar.file_uploader("Upload a TensorFlow Saved Model directory", type=["dir"])

# Main content
if uploaded_model:
    st.header("Model Conversion")

    # Load the model
    model = tf.saved_model.load(uploaded_model)

    # Button to initiate conversion
    if st.button("Convert to TensorFlow.js"):
        try:
            output_dir = st.text_input("Specify the output directory for TensorFlow.js model:")
            if output_dir:
                tfjs.converters.save_keras_model(model, output_dir)
                st.success("Conversion completed successfully!")
                st.markdown(f"The TensorFlow.js model is saved in the directory: {output_dir}")
            else:
                st.warning("Please specify the output directory.")
        except Exception as e:
            st.error(f"An error occurred during conversion: {str(e)}")

# Display some information
st.sidebar.info(
    "This tool allows you to convert a TensorFlow Saved Model into a TensorFlow.js model. "
    "Upload a directory containing the Saved Model, and click the 'Convert to TensorFlow.js' button."
)

# Footer
st.sidebar.text("Created by Your Name")

