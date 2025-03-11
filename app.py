import streamlit as st
import os
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.models import resnet50

# Define constants
IM_SIZE = 224

# Set page config
st.set_page_config(
    page_title="Vehicle Classifier",
    page_icon="🚗",
    layout="wide"
)

# Define transforms
@st.cache_resource
def get_transforms():
    return transforms.Compose([
        transforms.Resize((IM_SIZE, IM_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# Load model
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = resnet50()
    
    # Adjust the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 10)
    
    # Load saved weights
    model_path = os.path.join("best_model", "best_model_resnet50.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, device

# Get class names
@st.cache_data
def get_class_names():
    return ['SUV', 'bus', 'family sedan', 'fire engine', 'heavy truck', 
            'jeep', 'minibus', 'racing car', 'taxi', 'truck']

# Function to make predictions
def predict(image, model, device, transform):
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(outputs, 1).item()
    
    return predicted_class, probabilities.cpu().numpy()

# Main app
def main():
    st.title("🚗 Vehicle Classification System")
    
    st.markdown("""
    ## Welcome to the Vehicle Classifier!
    This application uses a deep learning model based on ResNet50 to classify vehicles into 10 different categories.
    
    ### How to use:
    1. Upload a vehicle image using the file uploader below
    2. Wait for the prediction to process
    3. View the predicted vehicle type and confidence scores
    """)
    
    # Initialize model
    with st.spinner("Loading model... This might take a few seconds."):
        model, device = load_model()
        transform = get_transforms()
        class_names = get_class_names()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a vehicle image...", type=["jpg", "jpeg", "png"])
    
    col1, col2 = st.columns(2)
    
    if uploaded_file is not None:
        with col1:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            with st.spinner("Analyzing image..."):
                # Make prediction
                predicted_idx, probabilities = predict(image, model, device, transform)
                
                # Display prediction
                st.success(f"### Prediction: {class_names[predicted_idx]}")
                
                # Display confidence scores
                st.subheader("Confidence Scores")
                
                # Create bar chart of probabilities
                fig, ax = plt.subplots(figsize=(10, 6))
                y_pos = np.arange(len(class_names))
                ax.barh(y_pos, probabilities, align='center')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(class_names)
                ax.invert_yaxis()  # labels read top-to-bottom
                ax.set_xlabel('Probability')
                ax.set_title('Prediction Probabilities')
                
                # Highlight the predicted class
                bars = ax.barh(y_pos, probabilities, align='center')
                bars[predicted_idx].set_color('green')
                
                st.pyplot(fig)

if __name__ == "__main__":
    main()