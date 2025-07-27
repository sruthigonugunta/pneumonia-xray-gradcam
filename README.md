# Pneumonia Detection from Chest X-rays (Streamlit App + Grad-CAM)
This project includes code for training a ResNet50 model to classify chest X-ray images as pneumonia or normal, and a Streamlit app to make predictions on uploaded images with Grad-CAM heatmap visualization.

Built as part of my interest in combining biomedical engineering and machine learning for medical diagnostics.

**Key Scripts:**

- pneumonia_model.py – trains a binary classifier using transfer learning (ResNet50), saves the model in both .keras and .h5 formats, and includes Grad-CAM code for testing.

- app.py – Streamlit web app that:
    - Loads a trained model (if available)
    - Accepts uploaded X-rays from users
    - Predicts class (PNEUMONIA or NORMAL)
    - Generates and overlays Grad-CAM heatmaps to show important regions

**Dataset & Model Info:**

- Dataset: Kaggle Chest X-ray Pneumonia dataset
- Model architecture: ResNet50 backbone with a custom classification head
- Training: 5 epochs using binary cross-entropy + Adam optimizer
- Image size: 224 × 224, normalized to [0, 1]

**How to Run the App:**

- Install dependencies:
    - pip install -r requirements.txt

- Launch the web app:
    - streamlit run app.py

- Upload any chest X-ray image (.jpg, .png) to get predictions and a heatmap.

**Notes:**

- If no model file is found, the app will run using a small fallback CNN (not trained).
- Grad-CAM visualizations are based on the last convolutional layer (conv5_block3_out in ResNet50).
- Heatmaps are generated using the tf-explain library and visualized with OpenCV.

**About Me:**

I am Sruthi Gonugunta, a Biomedical Engineering student at Georgia Tech with a CS minor. I am passionate about combining AI with healthcare, especially in diagnostics, imaging, and medical software.
www.linkedin.com/in/sruthi-gonugunta
