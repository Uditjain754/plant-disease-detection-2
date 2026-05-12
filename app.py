import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="wide"
)

# ======================================
# CUSTOM CSS
# ======================================

st.markdown("""
<style>

.main-header {
    font-size: 45px;
    color: #2E7D32;
    text-align: center;
    font-weight: bold;
}

.sub-header {
    font-size: 28px;
    color: #388E3C;
    margin-top: 20px;
}

.prediction-box {
    background-color: #E8F5E9;
    padding: 20px;
    border-radius: 10px;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# ======================================
# MODEL ARCHITECTURE
# ======================================

class PlantDiseaseModel(nn.Module):

    def __init__(self, num_classes):
        super(PlantDiseaseModel, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # SAME AS TRAINED MODEL
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        x = self.global_avg_pool(x)

        x = self.fc_block(x)

        return x

# ======================================
# LOAD MODEL
# ======================================

@st.cache_resource
def load_model_resources():

    # CONFIG
    with open("model_config.json", "r") as f:
        config = json.load(f)

    # CLASS NAMES
    with open("class_names.json", "r") as f:
        class_names = json.load(f)

    # LABEL ENCODER
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    # TRANSFORM
    with open("inference_transform.pkl", "rb") as f:
        transform = pickle.load(f)

    # DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MODEL
    model = PlantDiseaseModel(num_classes=len(class_names))

    checkpoint = torch.load(
        "best_model.pth",
        map_location=device
    )

    model.load_state_dict(
        checkpoint,
        strict=False
    )

    model.to(device)

    model.eval()

    return model, transform, label_encoder, class_names, device

# ======================================
# PREDICTION FUNCTION
# ======================================

def predict_image(image, model, transform, device, label_encoder):

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = label_encoder.inverse_transform(
        [predicted.item()]
    )[0]

    return predicted_class, confidence.item() * 100, probabilities.cpu().numpy()[0]

# ======================================
# DISEASE INFORMATION
# ======================================

disease_info = {

    "Pepper__bell___Bacterial_spot": {
        "description": "Bacterial spot causes dark lesions on pepper leaves and fruits.",
        "treatment": "Use copper fungicides and remove infected leaves."
    },

    "Pepper__bell___healthy": {
        "description": "Healthy pepper plant.",
        "treatment": "No treatment required."
    },

    "Potato___Early_blight": {
        "description": "Early blight creates brown concentric spots on leaves.",
        "treatment": "Apply fungicides regularly."
    },

    "Potato___Late_blight": {
        "description": "Late blight spreads rapidly and destroys leaves.",
        "treatment": "Destroy infected plants immediately."
    },

    "Potato___healthy": {
        "description": "Healthy potato plant.",
        "treatment": "No treatment required."
    },

    "Tomato_Bacterial_spot": {
        "description": "Bacterial spot affects tomato leaves and fruits.",
        "treatment": "Use disease-free seeds and copper sprays."
    },

    "Tomato_Early_blight": {
        "description": "Brown spots appear on lower leaves.",
        "treatment": "Use fungicides and crop rotation."
    },

    "Tomato_Late_blight": {
        "description": "Water-soaked lesions spread quickly.",
        "treatment": "Destroy infected plants."
    },

    "Tomato_Leaf_Mold": {
        "description": "Leaf mold creates yellow patches on leaves.",
        "treatment": "Improve ventilation."
    },

    "Tomato_Septoria_leaf_spot": {
        "description": "Small circular spots appear on leaves.",
        "treatment": "Use fungicides."
    },

    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "description": "Spider mites suck sap from leaves.",
        "treatment": "Use neem oil spray."
    },

    "Tomato__Target_Spot": {
        "description": "Dark target-like lesions appear.",
        "treatment": "Apply fungicides regularly."
    },

    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "description": "Leaves curl and become yellow.",
        "treatment": "Control whiteflies."
    },

    "Tomato__Tomato_mosaic_virus": {
        "description": "Mosaic patterns appear on leaves.",
        "treatment": "Remove infected plants."
    },

    "Tomato_healthy": {
        "description": "Healthy tomato plant.",
        "treatment": "No treatment required."
    }
}

# ======================================
# MAIN APP
# ======================================

def main():

    model, transform, label_encoder, class_names, device = load_model_resources()

    st.markdown(
        "<h1 class='main-header'>🌿 Plant Disease Classifier</h1>",
        unsafe_allow_html=True
    )

    st.write("Upload a plant leaf image to detect disease.")

    uploaded_file = st.file_uploader(
        "Choose Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        # IMAGE
        with col1:

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        # PREDICTION
        with col2:

            predicted_class, confidence, probs = predict_image(
                image,
                model,
                transform,
                device,
                label_encoder
            )

            formatted_name = predicted_class.replace("_", " ")

            st.markdown(
                f"""
                <div class='prediction-box'>
                    <h2>Prediction</h2>
                    <h3>{formatted_name}</h3>
                    <h4>Confidence: {confidence:.2f}%</h4>
                </div>
                """,
                unsafe_allow_html=True
            )

            # TOP PREDICTIONS

            top_indices = np.argsort(probs)[::-1][:5]

            top_classes = [
                label_encoder.inverse_transform([i])[0].replace("_", " ")
                for i in top_indices
            ]

            top_probs = [probs[i] * 100 for i in top_indices]

            fig, ax = plt.subplots(figsize=(8,4))

            ax.barh(top_classes, top_probs)

            ax.set_xlabel("Confidence %")

            ax.set_title("Top Predictions")

            st.pyplot(fig)

            # DISEASE INFO

            if predicted_class in disease_info:

                st.markdown(
                    "<h2 class='sub-header'>Disease Description</h2>",
                    unsafe_allow_html=True
                )

                st.write(
                    disease_info[predicted_class]["description"]
                )

                st.markdown(
                    "<h2 class='sub-header'>Treatment</h2>",
                    unsafe_allow_html=True
                )

                st.write(
                    disease_info[predicted_class]["treatment"]
                )

# ======================================
# RUN APP
# ======================================

if __name__ == "__main__":
    main()