import torch 
import numpy as np
import torch.nn as nn
from torch import cuda

device = "cuda" if cuda.is_available() else "cpu"
print(f"Using {device} device")

class Model(nn.Module):
    def __init__(self, no_layers = 2, embs_weigth = None, dimensions = None, no_of_classes = 4):
        super(Model,self).__init__()
        
        self.no_layers = no_layers
        self.dim = dimensions
        self.embeddings_weight = embs_weigth
        self.no_of_classes = no_of_classes
        self.embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(self.embeddings_weight).float())

        self.lstm1 = nn.LSTM(input_size = 100, hidden_size = self.dim,
                           num_layers = self.no_layers, batch_first=True,dropout = 0.5)
        
        self.lstm2 =  nn.LSTM(input_size = 100, hidden_size = self.dim,
                           num_layers = self.no_layers, batch_first=True, dropout = 0.5)
        
        self.fc1 = nn.Linear(self.dim, self.no_of_classes)
        self.relu = nn.ReLU(inplace = False)

        
    def forward(self, text, entity, hidden_state1):
        embeddings_text = self.embedding(text)
        embeddings_entity = self.embedding(entity)
        
        output_text, hidden_state1 = self.lstm1(embeddings_text, hidden_state1)
        _, output_entity = self.lstm2(embeddings_entity, hidden_state1)
        out = self.relu(output_entity[0][-1])
        out = self.fc1(out)
        
        return out
    