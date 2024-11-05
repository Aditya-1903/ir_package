import streamlit as st
import torch
import faiss
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Function to load the fine-tuned model
def load_finetuned_model(pth_file):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.load_state_dict(torch.load(pth_file, map_location=torch.device("cpu")))
    model.eval()
    return model

# Function for text query
def query_text_faiss(text_query, model, processor, index, metadata, k=5):
    inputs = processor(text=text_query, return_tensors="pt")

    with torch.no_grad():
        query_embedding = model.get_text_features(**inputs)
        query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
        query_embedding_np = query_embedding.cpu().numpy()

    D, I = index.search(query_embedding_np, k)
    closest_images = [(metadata[i], D[0][idx]) for idx, i in enumerate(I[0])]
    closest_images = sorted(closest_images, key=lambda x: x[1])

    return closest_images

# Function for image query
def query_image_faiss(query_image, uploaded_image, model, processor, index, metadata, k=5):
    inputs = processor(images=query_image, return_tensors="pt")

    with torch.no_grad():
        query_embedding = model.get_image_features(**inputs)
        query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
        query_embedding_np = query_embedding.cpu().numpy()

    D, I = index.search(query_embedding_np, k)
    
    closest_images = [(metadata[i], D[0][idx]) for idx, i in enumerate(I[0])]
    closest_images = sorted(closest_images, key=lambda x: x[1])

    filtered_images = [(path, score) for path, score in closest_images if not path.endswith(uploaded_image.name)]
    return filtered_images[:k]


# Load model, processor, index, and metadata
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = load_finetuned_model("clip_finetuned_model.pth")
index = faiss.read_index("faiss_index.index")
metadata = list(pd.read_csv("metadata.csv")["metadata"].values)

# Streamlit app
st.title("Closet Companion: Your Personalized Outfit Finder")
st.write("Use a text description or upload an image to find similar outfits.")

# User options for query type
query_type = st.radio("Choose your query type:", ("Text Query", "Image Query"))

top_k = 3

# Handle Text Query
if query_type == "Text Query":
    text_query = st.text_input("Enter your outfit description here: ")
    if st.button("Search by Text") and text_query:
        with st.spinner("Searching for similar outfits..."):
            closest_images = query_text_faiss(text_query, model, processor, index, metadata, k=top_k)
            for i, (image_path, score) in enumerate(closest_images, start=1):
                #st.write(f"Image {i} (Score: {score:.2f})")
                image = Image.open(image_path)
                st.image(image, width=150)

# Handle Image Query
elif query_type == "Image Query":
    uploaded_image = st.file_uploader("Upload an image of the outfit to find similar ones: ", type=["jpg", "png", "jpeg"])
    if uploaded_image is not None:
        query_image = Image.open(uploaded_image).convert("RGB")
        
        st.image(query_image, caption="Uploaded Image", width=200)
    
        if st.button("Search by Image"):
            with st.spinner("Searching for similar outfits..."):
                closest_images = query_image_faiss(query_image, uploaded_image, model, processor, index, metadata, k=top_k)
                for i, (image_path, score) in enumerate(closest_images, start=1):
                    #st.write(f"Image {i} (Score: {score:.2f})")    
                    image = Image.open(image_path)
                    st.image(image, width=150)