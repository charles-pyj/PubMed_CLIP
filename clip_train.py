import clip
import torch
from PIL import Image
import json
import time
import torch
import sys
from torch import nn
from torch.utils.data import Sampler
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from torch.utils.data.dataloader import default_collate
import numpy as np
import os
import requests
from tqdm import tqdm 
device='cuda' if torch.cuda.is_available() else 'cpu'
print(device)

def custom_collate_fn(batch):
    initial = len(batch)
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    if initial != len(batch):
        print("Collating")
    return default_collate(batch)

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, texts,  transform=None):
        """
        Args:
            annotations_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_labels = texts
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_dir)
    

    def __getitem__(self, idx):
        if get_image_url(self.img_dir[idx],idx):
            try:
               visual = self.transform(Image.open(f"./tmp/{idx}.jpg")).to(device)
               text_embed = clip.tokenize(self.img_labels[idx][:77],context_length=77).squeeze().to(device)
               os.remove(f"./tmp/{idx}.jpg")
               return visual, text_embed
            except : 
                print(f"Skipping image at index {idx} due to UnidentifiedImageError")
                os.remove(f"./tmp/{idx}.jpg")
                return None, None
        # image = preprocess(read_image(self.img_dir[idx], mode=ImageReadMode.RGB))
        # try:
        #     visual = self.transform(Image.open(f"./tmp/{idx}.jpg")).to(device)
        #     text_embed = clip.tokenize(self.img_labels[idx][:77],context_length=77).squeeze().to(device)
        #     return visual, text_embed
        # except :
        #     print(f"Skipping image at index {idx} due to UnidentifiedImageError")
        #     return None, None


def read_json(path):
    with open(path,'r') as f:
        data = json.load(f)
    return data

def get_optimizer(params,lr=1e-5,weight_decay=4e-4):
    optimizer = torch.optim.Adam(
        params,
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    )
    return optimizer

def load_model(checkpoint_path):
    model, preprocess = clip.load("ViT-L/14", device=device)
    #checkpoint = torch.load(checkpoint_path)
    #model.load_state_dict(checkpoint)
    model.train()
    model.to('cuda')
    model.float()
    params = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            params.append(p)  # Only append the parameter itself
    return model, preprocess, params

def train(model,train_loader,val_loader,optimizer,batch_size):
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    model.train()
    model = model.to('cuda')
    for epoch in range(1):
        model.train()
        print(f"Epoch {epoch}")
        train_all_loss = 0
        number_batch = len(train_loader)
        print(f"Number of batch: {number_batch}")
        # Predicting and computing score
        for i, (image, labels) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}", unit="batch")):
            if image is None or labels is None:
                continue  # Skip this batch
            print(f"Batch {i}")
            optimizer.zero_grad()
            images = torch.stack([img for img in image], dim=0).to(device)
            #print(images.dtype)
            size = images.shape[0]
            logits_per_image, logits_per_text = model(images, labels)
            logits_per_image *= (np.exp(0.01) / np.exp(0.07))
            logits_per_text *= (np.exp(0.01) / np.exp(0.07))

            ground_truth = torch.arange(size, dtype=torch.long, device=device)
            lambdaa = 0.5
            train_total_loss = lambdaa*(loss_img(logits_per_image, ground_truth)) + (1-lambdaa)* (loss_txt(logits_per_text, ground_truth))
            #print(f"loss: {train_total_loss}")
            train_total_loss.backward()
            optimizer.step()
            #clip.model.convert_weights(model)
            train_all_loss += train_total_loss
        #evaluate(model,val_loader)
        print(f"Average loss over epoch: {train_all_loss}")
    torch.save(model, "./checkpoint_test.pth")
    print(f"Model saved to ./checkpoint_ViT.pth")

def evaluate(model, val_loader):
    model.eval()
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    train_all_loss = 0

    with torch.no_grad():  # Ensure gradients are not computed
        for i, (image, labels) in enumerate(val_loader):
            print(f"Batch {i}")
            images = torch.stack([img for img in image], dim=0).to(device)
            size = images.shape[0]
            # Get the model outputs
            logits_per_image, logits_per_text = model(images, labels)
            
            # Scaling logits
            logits_per_image *= (np.exp(0.01) / np.exp(0.07))
            logits_per_text *= (np.exp(0.01) / np.exp(0.07))
            
            # Compute probabilities
            probabilities = F.softmax(logits_per_image, dim=1)
            
            # Ground truth
            ground_truth = torch.arange(size, dtype=torch.long, device=device)
            
            # Compute loss
            lambdaa = 0.5
            train_total_loss = lambdaa * loss_img(logits_per_image, ground_truth) + (1 - lambdaa) * loss_txt(logits_per_text, ground_truth)
            
            # Accumulate the total loss
            train_all_loss += train_total_loss.item()  # Use .item() to avoid keeping the computation graph
            
            # Display choices
            print("Choices")
            for prob in probabilities:
                choice = torch.argmax(prob)
                print(choice)
        
    print(f"Average evaluation loss: {train_all_loss / len(val_loader)}")

def get_image_url(url,idx):
    responseImg = requests.get(url)
    if responseImg.status_code == 200:
        #print("Success!")
        image_data = responseImg.content
        with open(f'./tmp/{idx}.jpg', 'wb') as file:
            file.write(image_data)
            return True
    return False


def get_pmc_image(records_path,img_path):
    records = read_json(records_path)
    dirs = os.listdir(img_path)
    sorted_dirs = sorted(dirs, key=lambda x: int(x.split('.')[0]))
    img_paths = [os.path.join(img_path,i) for i in sorted_dirs]
    captions = [i['caption'] for i in records]
    return img_paths, captions

if __name__ == "__main__":
    #records = read_json("/nfs/turbo/umms-drjieliu/proj/medlineKG/data/figure_json_by_article/test/test_caption.json")
    # model = torch.load("./checkpoint_new.pth")
    # torch.save(model.state_dict(),"./CLIP_state_new.pth")
    model, preprocess, trainable_params = load_model("./CLIP_state_new.pth")

    # records_path = "/nfs/turbo/umms-drjieliu/proj/medlineKG/data/figure_json_by_article/test/test_caption.json"
    # img_path = "../clip_img"
    # img_paths, captions = get_pmc_image(records_path,img_path)
    img_paths = read_json("/nfs/turbo/umms-drjieliu/proj/medlineKG/data/figure_json_by_article/pmcimage_paths.json")[:100]
    captions = read_json("/nfs/turbo/umms-drjieliu/proj/medlineKG/data/figure_json_by_article/pmcimage_captions.json")[:100]
    train_set = CustomImageDataset(
    img_dir=img_paths,texts=captions,transform=preprocess)
    #print(img_paths[900:])
    val_set = CustomImageDataset(img_dir=img_paths[:100],texts=captions[:100],transform=preprocess)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=4,
        collate_fn = custom_collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=8,
        collate_fn = custom_collate_fn
    )
    train(model,train_loader,test_loader,get_optimizer(trainable_params,1e-5),batch_size=32)
    evaluate(model,test_loader)

