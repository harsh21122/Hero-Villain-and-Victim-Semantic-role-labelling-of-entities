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

# Download with: python -m spacy download en
spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary:
    def __init__(self, vocab, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<UNK>" : 1}
        self.freq_threshold = freq_threshold
        self.vocab = vocab
        
    def __len__(self):
        return len(self.stoi)
    
    def preprocess(self, text):    
        print
        text = text.replace(" ' ", "'")
        text = text.replace(" â€™ ", 'â€™')
        text = re.sub(r'\w*\d\w*', '', text)
        text = text.replace("\n", " ")
        text = text.replace("\\", " ")
        for itr in string.punctuation:
                text = text.replace(itr, ' ')
        return text    
    
    def tokenizer_eng(self, text):
        tokenized = []
        text = self.preprocess(text)
        for tok in spacy_eng.tokenizer(text):
            for word in spacy_eng.tokenizer(contractions.fix(tok.text.lower())):
                tokenized.append(word.text.lower())
        return tokenized

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    try:
                        idx = self.vocab.index(word)
                        self.stoi[word] = idx
                        self.itos[idx] = word
                    except ValueError:
                        pass
                    

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        numericalized_token = []
        
        for token in tokenized_text:
            if token in self.stoi:
                numericalized_token.append(self.stoi[token])
            else:
                numericalized_token.append(self.stoi["<UNK>"])
                
        return numericalized_token

class Custom(Dataset):
    def __init__(self, root, captions_file, vocab = None, transform = None, freq_threshold = 5, roletolabel = {}):
        self.df = pd.read_csv(captions_file, index_col=None)
        self.transform = transform
        self.roletolabel = roletolabel
        self.root = root
        
        # Getting text, role, entities and images name columns
        self.texts = self.df["Text"]
        self.entities = self.df["Entity"]
        self.roles = self.df["Role"]
        self.images = self.df["Name"]

        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(vocab, freq_threshold)
        self.vocab.build_vocabulary(self.texts.tolist())
        self.vocab.build_vocabulary(self.entities.tolist())
        

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        role = self.roletolabel[self.roles[index]]
        text = self.texts[index] 
        entity = self.entities[index]
        image = Image.open(os.path.join(self.root, self.images[index]))
        
        if self.transform:
            image = self.transform(image)
            
        text_vec = []
        entity_vec = []
        text_vec.extend(self.vocab.numericalize(text))
        entity_vec.extend(self.vocab.numericalize(entity))
        
        return torch.Tensor(text_vec).int(), image, torch.Tensor(entity_vec).int(), torch.tensor(role).int()

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        text =  []
        entity = []
        image = []
        role = []
        for item in batch:
            text.append(item[0])
            image.append(item[1])      
            entity.append(item[2])
            role.append(item[3])
            
        text = pad_sequence(text, batch_first=True, padding_value=self.pad_idx)
        entity = pad_sequence(entity, batch_first=True, padding_value=self.pad_idx)
        images = torch.stack(image)
        roles = torch.stack(role)
        
        return text, images, entity, roles