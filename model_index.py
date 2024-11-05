import torch
import faiss
import pandas as pd
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class ImageDataset(Dataset):
    def __init__(self, image_dir, processor):
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.jpg', '.png', '.jpeg'))]
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].squeeze(0), self.image_paths[idx]



def load_finetuned_model(pth_file):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.load_state_dict(torch.load(pth_file, map_location = torch.device('cpu')))
    model.eval()
    return model


def compute_and_store_embeddings(image_dir, model, processor, faiss_index_path):
    dataset = ImageDataset(image_dir, processor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    d = 512 
    index = faiss.IndexFlatL2(d)
    metadata = []

    with torch.no_grad():
        for images, image_paths in tqdm(dataloader):
            embeddings = model.get_image_features(pixel_values=images)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # Normalize
            embeddings_np = embeddings.cpu().numpy()

            index.add(embeddings_np)
            metadata.extend(image_paths)
    
    metadata_df = pd.DataFrame({"metadata":metadata})
    metadata_df.to_csv("metadata.csv",index=False)

    faiss.write_index(index, faiss_index_path)

    return index, metadata


# image query
def query_image_faiss(query_image_path, model, processor, index, metadata, k=5):
    query_image = Image.open(query_image_path).convert("RGB")
    inputs = processor(images=query_image, return_tensors="pt")

    with torch.no_grad():
        query_embedding = model.get_image_features(**inputs)
        query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
        query_embedding_np = query_embedding.cpu().numpy()

  
    D, I = index.search(query_embedding_np, k)  
    closest_image_path = metadata[I[0][0]]
    return closest_image_path

# text query
def query_text_faiss(text_query, model, processor, index, metadata, k=5):
    inputs = processor(text=text_query, return_tensors="pt")

    with torch.no_grad():
        query_embedding = model.get_text_features(**inputs)
        query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
        query_embedding_np = query_embedding.cpu().numpy()

  
    D, I = index.search(query_embedding_np, k)
    
    print(D)

    closest_image_path = metadata[I[0][0]]
    
    return closest_image_path

def main():
 
    image_dir = "input_images\\images"
    query_image_path = "input_images\\images\\1531.jpg"
    fine_tuned_model_path = "clip_finetuned_model.pth"
    faiss_index_path = "faiss_index.index"
    text_query="red polo tshirt"

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = load_finetuned_model(fine_tuned_model_path)

    
    # index, metadata = compute_and_store_embeddings(image_dir, model, processor, faiss_index_path)
    index = faiss.read_index(faiss_index_path)
    metadata = list(pd.read_csv('metadata.csv')['metadata'].values)

    closest_image_text = query_text_faiss(text_query, model, processor, index, metadata)
    #closest_image_image = query_image_faiss(query_image_path, model, processor, index, metadata)

    print(f"The most similar image is located at: {closest_image_text}")


if __name__ == "__main__":
    main()
