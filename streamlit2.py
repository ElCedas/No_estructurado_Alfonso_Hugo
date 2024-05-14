import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
import os
from lime import lime_image
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import io

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_checkpoint(filename, model, optimizer):
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    st.write("Checkpoint loaded successfully.")

def initialize_model(num_classes):
    model = models.shufflenet_v2_x1_0(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return model, optimizer

def get_class_names(data_dir):
    class_names = [d for d in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, d))]
    return class_names

def predict_image(images, model, class_names):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if not isinstance(images, list):
        images = [images]  # Convert single image to list for uniform processing

    images = [transform(img).unsqueeze(0) for img in images]
    images = torch.cat(images, dim=0)
    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, predicted_indices = torch.max(outputs, 1)
        predicted_labels = [class_names[idx] for idx in predicted_indices]

    return predicted_labels

def lime_explanation(model, image, class_names):
    explainer = lime_image.LimeImageExplainer()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    def batch_predict(images):
        images = torch.stack([transform(Image.fromarray(img.astype('uint8'))) for img in images])
        images = images.to(device)
        model.eval()
        with torch.no_grad():
            preds = model(images)
        return preds.cpu().numpy()

    explanation = explainer.explain_instance(np.array(image.convert('RGB')),
                                             classifier_fn=batch_predict,
                                             top_labels=1,
                                             hide_color=0,
                                             num_samples=1000,
                                             num_features=5)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, hide_rest=False)
    
    # Create a heatmap from the mask and blend it with the original image
    plt.figure(figsize=(8, 8))
    plt.imshow(np.array(image), interpolation='none')
    plt.imshow(mask, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.tight_layout()

    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_boundaries = Image.open(buf)

    plt.close()

    return np.array(img_boundaries)

# Streamlit UI
st.title('Image Classification with LIME Explanation')

data_dir = './data/pets/formatted_images'  # Adjust the path to your dataset
class_names = get_class_names(data_dir)
num_classes = len(class_names)
model, optimizer = initialize_model(num_classes)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    if st.button('Predict'):
        load_checkpoint('Shuffle_net_model_checkpoint.pth', model, optimizer)
        predicted_labels = predict_image([image], model, class_names)
        st.write('Predicted Class:', predicted_labels[0])
        st.write('Esperar un momento para explicacion de LIME debajo')
        img_boundaries = lime_explanation(model, image, class_names)
        st.image(img_boundaries, caption='LIME Explanation', use_column_width=True)
