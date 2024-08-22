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
import random
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt

device='cuda' if torch.cuda.is_available() else 'cpu'
print(device)

def get_image_tmp(url):
    responseImg = requests.get(url)
    if responseImg.status_code == 200:
        #print("Success!")
        image_data = responseImg.content
        with open(f'./tmp.jpg', 'wb') as file:
            file.write(image_data)
            return True
    return False

def get_visual_embed(model,path):
    image = Image.open(path)
    return model.visual(torch.stack([preprocess(image).to(device)]))


def get_text_embed(model,string):
    tokens = clip.tokenize(string,context_length=77).to(device)
    return model.encode_text(tokens).to(device)

def get_similarity(model,url,caption):
    get_image_tmp(url)
    visual = get_visual_embed(model,"./tmp.jpg")
    text_embed = get_text_embed(model,caption)
    #print(visual.shape)
    #print(text_embed.shape)
    return F.cosine_similarity(visual,text_embed,dim=1).item()

def shuffle_list(img_path,captions):
    combined = list(zip(img_path, captions))
    random.shuffle(combined)

    # Separate the combined list back into two lists
    shuffled_img_paths, shuffled_captions = zip(*combined)

    # If you need the shuffled results back in list form
    shuffled_img_paths = list(shuffled_img_paths)
    shuffled_captions = list(shuffled_captions)
    return shuffled_img_paths,shuffled_captions

def custom_collate_fn(batch):
    # Filter out invalid entries
    batch = list(filter(lambda x: x is not None and x[0] is not None and x[1] is not None, batch))
    if len(batch) == 0:
        return None,None
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
               text_embed = clip.tokenize(self.img_labels[idx],context_length=77).squeeze().to(device)
               os.remove(f"./tmp/{idx}.jpg")
               return visual, text_embed
            except : 
                print(f"Skipping image at index {self.img_dir[idx]} due to UnidentifiedImageError")
                os.remove(f"./tmp/{idx}.jpg")
                return None, None
        else :
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
    model, preprocess = clip.load("RN50", device=device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model.train()
    model.to('cuda')
    model.float()
    params = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            params.append(p)  # Only append the parameter itself
    return model, preprocess, params

def load_model_checkpoint(checkpoint_path,base_model):
    print(f"Model loaded from {checkpoint_path}")
    model, preprocess = clip.load(base_model, device=device)
    model = torch.load(checkpoint_path)
    # checkpoint = torch.load(checkpoint_path)
    # model.load_state_dict(checkpoint)
    model.train()
    model.to('cuda')
    model.float()
    params = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            params.append(p)  # Only append the parameter itself
    return model, preprocess, params

def load_model_raw():
    print(f"Model loaded from raw")
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

def recall_at_k(probs, true_labels, k=1):
    correct = 0
    top_k_preds = probs.topk(k, dim=1).indices
    for i, label in enumerate(true_labels):
        if label in top_k_preds[i]:
            correct += 1
    return correct / len(true_labels)

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
    torch.save(model, "./checkpoint_ViT_short.pth")
    print(f"Model saved!")

def evaluate(model, val_loader):
    model.eval()
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    train_all_loss = 0
    recall1 = []
    recall5 = []
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
            recall1.append(recall_at_k(probabilities,ground_truth,k=1))
            recall5.append(recall_at_k(probabilities,ground_truth,k=min(5,size)))
            # Compute loss
            lambdaa = 0.5
            train_total_loss = lambdaa * loss_img(logits_per_image, ground_truth) + (1 - lambdaa) * loss_txt(logits_per_text, ground_truth)
            
            # Accumulate the total loss
            train_all_loss += train_total_loss.item()  # Use .item() to avoid keeping the computation graph
            
            # Display choices
            # print("Choices")
            # for prob in probabilities:
            #     choice = torch.argmax(prob)
            #     print(choice)
        
    print(f"Average evaluation loss: {train_all_loss / len(val_loader)}")
    print(f"Recall @ 1: {np.mean(recall1)}")
    print(f"Recall @ 5: {np.mean(recall5)}")

def get_image_url(url,idx):
    #print(url)
    try: 
        responseImg = requests.get(url)
        if responseImg.status_code == 200:
            #print("Success!")
            image_data = responseImg.content
            with open(f'./tmp/{idx}.jpg', 'wb') as file:
                file.write(image_data)
                return True
        else: 
            return False
    except: 
        print(f"Request error at: {idx}")
        return False


def get_pmc_image(records_path,img_path):
    records = read_json(records_path)
    dirs = os.listdir(img_path)
    sorted_dirs = sorted(dirs, key=lambda x: int(x.split('.')[0]))
    img_paths = [os.path.join(img_path,i) for i in sorted_dirs]
    captions = [i['caption'] for i in records]
    return img_paths, captions

def try_tokenize(texts):
    token_sizes = []
    length_cnt = 0
    for i in tqdm(range(len(texts))):
        try:
            text_embed = clip.tokenize(texts[i],context_length=1024).squeeze().to(device)
            #print(text_embed.shape)
            non_padding_tokens = (text_embed != 0).sum().item()
            token_sizes.append(non_padding_tokens)
            #print(f"Num of tokens: {non_padding_tokens}")
        except: 
            #print("limit exceeded")
            length_cnt += 1
            token_sizes.append(10000)
    print(f"Ratio: {length_cnt/len(texts)}")
    with open("./token_length.json","w") as f:
        json.dump(token_sizes,f)
    # sns.kdeplot(token_sizes, fill=True, color="skyblue", bw_adjust=0.5)
    # plt.xlabel('Token Cnt')
    # plt.ylabel('Density')
    # plt.axvline(77, color='red', linestyle='--', linewidth=2)
    # plt.title('Multi Panel Distribution Density Plot')
    # # Save the plot
    # plt.savefig('./multiple_distribution_density_plot.png', dpi=300, bbox_inches='tight')
    # Display the plot
    

if __name__ == "__main__":
    #records = read_json("/nfs/turbo/umms-drjieliu/proj/medlineKG/data/figure_json_by_article/test/test_caption.json")
    # model = torch.load("./checkpoint_new.pth")
    # torch.save(model.state_dict(),"./CLIP_state_new.pth")
    model_raw, _, _ = load_model_raw()
    model_old, _, _ = load_model_checkpoint("/home/panyijun/PubMed_CLIP/checkpoint_ViT_new_0803.pth","ViT-L/14")
    model, preprocess, trainable_params = load_model_checkpoint("./checkpoint_ViT_short.pth","ViT-L/14")
    print(len(trainable_params))
    # records_path = "/nfs/turbo/umms-drjieliu/proj/medlineKG/data/figure_json_by_article/test/test_caption.json"
    # img_path = "../clip_img"
    # img_paths, captions = get_pmc_image(records_path,img_path)
    img_paths = read_json("/nfs/turbo/umms-drjieliu/proj/medlineKG/data/figure_json_by_article/pmcimage_paths_short.json")[-5000:]
    captions = read_json("/nfs/turbo/umms-drjieliu/proj/medlineKG/data/figure_json_by_article/pmcimage_captions_short.json")[-5000:]
    # json_test = read_json("/nfs/turbo/umms-drjieliu/proj/medlineKG/data/PMC_figure/test_set/test.json")
    # img_paths = [i[1] for i in json_test]
    # captions = [i[2] for i in json_test]

    
    img_paths, captions = shuffle_list(img_paths,captions)
    train_set = CustomImageDataset(
    img_dir=img_paths,texts=captions,transform=preprocess)
    #print(img_paths[900:])
    val_set = CustomImageDataset(img_dir=img_paths[:1000],texts=captions[:1000],transform=preprocess)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=32,
        collate_fn = custom_collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=128,
        collate_fn = custom_collate_fn
    )
    #print(len(captions))
    #try_tokenize(captions)
    # single panel: 0.314
    # multi panel: 
    #train(model,train_loader,test_loader,get_optimizer(trainable_params,1e-5),batch_size=32)
    print(f"Base model:")
    evaluate(model_raw,test_loader)
    print(f"Old model: ")
    evaluate(model_old,test_loader)
    print(f"New model: ")
    evaluate(model,test_loader)
    # similarity = []
    # print(img_paths[0])
    # for i in range(1000):
    #     similarity.append(get_similarity(model,img_paths[0],captions[i]))
    # print(similarity[0])
    # print(np.max(similarity[1:]))
    # print(img_paths[np.argmax(similarity[1:])])

    # print(captions[0][:77])
    # caption_similarity = []
    # for i in range(1000):
    #     caption_similarity.append(get_similarity(model,img_paths[i],captions[0]))
    # print(caption_similarity[0])
    # print(np.max(caption_similarity[1:]))
    # print(img_paths[np.argmax(caption_similarity[1:])])

