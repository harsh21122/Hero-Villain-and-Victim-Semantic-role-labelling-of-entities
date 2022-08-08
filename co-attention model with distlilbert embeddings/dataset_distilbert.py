import pandas as pd
import os
from torch.utils.data import Dataset
from PIL import Image
import spacy  
import torch
from torch.nn.utils.rnn import pad_sequence  
import string
import re
import contractions
import torchvision.transforms as transforms
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig

class Custom_emb(Dataset):
  def __init__(self, root, captions_file, transform = None, max_length=100, roletolabel = {}):
    self.df = pd.read_csv(captions_file, index_col=None)
    self.transform = transform
    self.roletolabel = roletolabel
    self.root = root
    self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    self.max_length = max_length
        
        # Getting text, role, entities and images name columns
    self.texts = self.df["Text"]
    self.entities = self.df["Entity"]
    self.roles = self.df["Role"]
    self.images = self.df["Name"]

  def __len__(self):
    return len(self.df)

  def __getitem__(self, index):
    entity = self.tokenizer.encode_plus(
            self.entities[index], 
            None,
            add_special_tokens=True,
            max_length = self.max_length,
            padding = True,
            truncation = True
        )
      
    text = self.tokenizer.encode_plus(
            self.texts[index], 
            None,
            add_special_tokens=True,
            max_length = self.max_length,
            padding = True,
            truncation = True
        )
    
    entity_id = entity["input_ids"]
    entity_mask = entity['attention_mask']

    padding_length1 = self.max_length - len(entity_id)
    entity_id = entity_id + ([0] * padding_length1)
    entity_mask = entity_mask + ([0] * padding_length1)

    text_id = text["input_ids"]
    text_mask = text['attention_mask']
    
  
    padding_length2 = self.max_length - len(text_id)
    text_id = text_id + ([0] * padding_length2)        
    text_mask = text_mask + ([0] * padding_length2)        
  

    role = self.roletolabel[self.roles[index]]
        
    image = Image.open(os.path.join(self.root, self.images[index]))
        
    if self.transform:
      image = self.transform(image)
            
    text_vec = []
    entity_vec = []
        
    return{
            'input_ids': torch.tensor(entity_id, dtype=torch.long),
            'attention_mask': torch.tensor(entity_mask, dtype=torch.long),
        },{
            'input_ids': torch.tensor(text_id, dtype=torch.long),
            'attention_mask': torch.tensor(text_mask, dtype=torch.long)}, image, torch.tensor(role).int()
