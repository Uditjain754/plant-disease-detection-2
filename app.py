import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import base64

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-header{
    font-size:55px;
    color:#1B5E20;
    text-align:center;
    font-weight:bold;
}

.sub-text{
    text-align:center;
    font-size:20px;
    color:#555;
    margin-bottom:30px;
}

.prediction-box{
    background-color:#E8F5E9;
    padding:25px;
    border-radius:15px;
    border:1px solid #C8E6C9;
}

.info-box{
    padding:20px;
    border-radius:15px;
    margin-bottom:20px;
}

.sidebar .sidebar-content{
    background-color:#F1F8E9;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# MODEL ARCHITECTURE
# =========================================================

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


# =========================================================
# BACKGROUND IMAGE
# =========================================================

def get_base64(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_image = get_base64("bg.jpg")

page_bg_img = f"""
<style>

[data-testid="stAppViewContainer"] {{
    background-image:
    linear-gradient(
        rgba(255,255,255,0.5),
        rgba(255,255,255,0.5)
    ),
    url("data:image/jpg;base64,{bg_image}");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}

[data-testid="stSidebar"] {{
    background: rgba(255,255,255,0.85);
}}

</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model_resources():

    with open("class_names.json", "r") as f:
        class_names = json.load(f)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    with open("inference_transform.pkl", "rb") as f:
        transform = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PlantDiseaseModel(num_classes=len(class_names))

    checkpoint = torch.load(
        "best_model.pth",
        map_location=device
    )

    model.load_state_dict(checkpoint, strict=False)

    model.to(device)

    model.eval()

    return model, transform, label_encoder, class_names, device

# =========================================================
# PREDICTION FUNCTION
# =========================================================

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

# =========================================================
# DISEASE INFORMATION
# =========================================================

disease_info = {

    "Pepper__bell___Bacterial_spot": {
        "description": "Bacterial spot causes dark, water-soaked lesions on leaves and fruit.",
        "causes": "Caused by Xanthomonas bacteria.",
        "treatment": "Apply copper-based bactericides.",
        "prevention": "Use disease-free seeds and avoid overhead watering."
    },

    "Pepper__bell___healthy": {
        "description": "Healthy pepper plant with vibrant green leaves.",
        "causes": "Proper plant care and nutrition.",
        "treatment": "No treatment required.",
        "prevention": "Maintain regular watering and monitoring."
    },

    "Potato___Early_blight": {
        "description": "Brown concentric lesions appear on leaves.",
        "causes": "Caused by Alternaria solani fungus.",
        "treatment": "Apply fungicides regularly.",
        "prevention": "Rotate crops and ensure proper spacing."
    },

    "Potato___Late_blight": {
        "description": "Water-soaked lesions spread rapidly.",
        "causes": "Caused by Phytophthora infestans.",
        "treatment": "Remove infected plants immediately.",
        "prevention": "Avoid excess moisture and use resistant varieties."
    },

    "Potato___healthy": {
        "description": "Healthy potato plant with lush foliage.",
        "causes": "Good farming practices.",
        "treatment": "No treatment required.",
        "prevention": "Maintain healthy soil conditions."
    },

    "Tomato_Bacterial_spot": {
        "description": "Small dark lesions appear on leaves and fruit.",
        "causes": "Bacterial infection through contaminated water.",
        "treatment": "Use copper-based sprays.",
        "prevention": "Avoid overhead irrigation."
    },

    "Tomato_Early_blight": {
        "description": "Brown spots with concentric rings.",
        "causes": "Alternaria fungus infection.",
        "treatment": "Apply fungicides.",
        "prevention": "Crop rotation and proper spacing."
    },

    "Tomato_Late_blight": {
        "description": "Rapid spread of dark lesions on leaves.",
        "causes": "Phytophthora infestans fungus.",
        "treatment": "Destroy infected plants.",
        "prevention": "Ensure proper air circulation."
    },

    "Tomato_Leaf_Mold": {
        "description": "Yellow patches appear on leaf surfaces.",
        "causes": "High humidity conditions.",
        "treatment": "Use fungicides.",
        "prevention": "Improve greenhouse ventilation."
    },

    "Tomato_Septoria_leaf_spot": {
        "description": "Small dark lesions with yellow halo.",
        "causes": "Septoria fungus thriving in wet conditions.",
        "treatment": "Remove infected leaves and spray fungicide.",
        "prevention": "Avoid wet foliage and rotate crops."
    },

    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "description": "Leaves become yellow and stippled.",
        "causes": "Spider mite infestation.",
        "treatment": "Use neem oil spray.",
        "prevention": "Maintain humidity and monitor plants."
    },

    "Tomato__Target_Spot": {
        "description": "Dark circular target-like spots appear.",
        "causes": "Fungal infection in humid conditions.",
        "treatment": "Apply fungicides regularly.",
        "prevention": "Keep leaves dry and prune infected parts."
    },

    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "description": "Leaves curl and turn yellow.",
        "causes": "Virus spread by whiteflies.",
        "treatment": "Control whitefly population.",
        "prevention": "Use resistant varieties and insect nets."
    },

    "Tomato__Tomato_mosaic_virus": {
        "description": "Mosaic patterns appear on leaves.",
        "causes": "Virus spread through contaminated tools.",
        "treatment": "Remove infected plants.",
        "prevention": "Sanitize tools and use healthy seeds."
    },

    "Tomato_healthy": {
        "description": "Healthy tomato plant with strong leaves.",
        "causes": "Balanced nutrition and proper care.",
        "treatment": "No treatment required.",
        "prevention": "Regular monitoring and watering."
    }
}

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🌿 About the Model")

st.sidebar.write(
    "This application uses a Convolutional Neural Network (CNN) "
    "to classify plant diseases from leaf images."
)

st.sidebar.header("Model Architecture")

st.sidebar.markdown("""
- CNN Based Deep Learning Model
- 15 Plant Disease Classes
- Trained on PlantVillage Dataset
- Real-Time Disease Prediction
""")

st.sidebar.header("Model Accuracy")

st.sidebar.success("Training Accuracy: 99.3%")

st.sidebar.header("Dataset Information")

st.sidebar.markdown("""
- PlantVillage Dataset
- Tomato, Potato, Pepper Leaves
- Healthy and Diseased Classes
""")

st.sidebar.header("Available Plant Diseases")

disease_list = [x.replace("_", " ") for x in disease_info.keys()]

selected_disease = st.sidebar.selectbox(
    "Available disease in our dataset",
    disease_list
)

st.sidebar.header("How To Use")

st.sidebar.markdown("""
1. Upload a plant leaf image  
2. Wait for prediction  
3. View confidence score  
4. Read disease information
""")

st.sidebar.header("Developer")

st.sidebar.success("UDIT JAIN")

# =========================================================
# MAIN APP
# =========================================================

def main():

    model, transform, label_encoder, class_names, device = load_model_resources()

    st.markdown(
        "<h1 class='main-header'>🌿 Plant Disease Classifier</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p class='sub-text'>AI Powered Plant Leaf Disease Detection using Deep Learning</p>",
        unsafe_allow_html=True
    )

    st.info(
        "Upload a plant leaf image and the model will predict the disease."
    )

    uploaded_file = st.file_uploader(
        "Upload Plant Leaf Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.markdown("## 🌱 Leaf Image Preview")

        st.image(
            image,
            caption="Uploaded Leaf Image",
            use_container_width=True
        )

        predicted_class, confidence, probs = predict_image(
            image,
            model,
            transform,
            device,
            label_encoder
        )

        formatted_name = predicted_class.replace("_", " ")

        st.markdown("## 📊 Prediction Result")

        st.markdown(
            f"""
            <div class='prediction-box'>
                <h2 style='color:#1B5E20;'>{formatted_name}</h2>
                <h3>Confidence: {confidence:.2f}%</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        # =========================================================
        # TOP PREDICTIONS GRAPH
        # =========================================================

        top_indices = np.argsort(probs)[::-1][:5]

        top_classes = [
            label_encoder.inverse_transform([i])[0].replace("_", " ")
            for i in top_indices
        ]

        top_probs = [probs[i] * 100 for i in top_indices]

        st.markdown("## 📈 Top Predictions")

        fig, ax = plt.subplots(figsize=(8,4))

        ax.barh(top_classes, top_probs)

        ax.set_xlabel("Confidence %")

        ax.set_title("Top Predictions")

        ax.invert_yaxis()

        st.pyplot(fig)

        # =========================================================
        # DISEASE INFORMATION SECTION
        # =========================================================

        if predicted_class in disease_info:

            info = disease_info[predicted_class]

            st.markdown("<br>", unsafe_allow_html=True)

            row1_col1, row1_col2 = st.columns(2)

            with row1_col1:

                st.markdown(
                    """
                    <div style="
                        background-color:#E8F5E9;
                        padding:25px;
                        border-radius:15px;
                        margin-bottom:20px;
                    ">
                    <h2 style="color:#1B5E20;">🩺 Disease Description</h2>
                    """,
                    unsafe_allow_html=True
                )

                st.write(info["description"])

                st.markdown("</div>", unsafe_allow_html=True)

            with row1_col2:

                st.markdown(
                    """
                    <div style="
                        background-color:#FFF8E1;
                        padding:25px;
                        border-radius:15px;
                        margin-bottom:20px;
                    ">
                    <h2 style="color:#E65100;">💊 Treatment</h2>
                    """,
                    unsafe_allow_html=True
                )

                st.write(info["treatment"])

                st.markdown("</div>", unsafe_allow_html=True)

            row2_col1, row2_col2 = st.columns(2)

            with row2_col1:

                st.markdown(
                    """
                    <div style="
                        background-color:#E3F2FD;
                        padding:25px;
                        border-radius:15px;
                    ">
                    <h2 style="color:#0D47A1;">⚠ Causes</h2>
                    """,
                    unsafe_allow_html=True
                )

                st.write(info["causes"])

                st.markdown("</div>", unsafe_allow_html=True)

            with row2_col2:

                st.markdown(
                    """
                    <div style="
                        background-color:#E8F5E9;
                        padding:25px;
                        border-radius:15px;
                    ">
                    <h2 style="color:#1B5E20;">🛡 Prevention</h2>
                    """,
                    unsafe_allow_html=True
                )

                st.write(info["prevention"])

                st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# RUN APP
# =========================================================

if __name__ == "__main__":
    main()
