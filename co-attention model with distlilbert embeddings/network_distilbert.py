from torch import cuda, dropout  
import torch
import torch.nn as nn

device = "cuda" if cuda.is_available() else "cpu"
print(f"Using {device} device")

seed = 42
if device == "cuda":
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed_all(seed) 


class ocr_Encoder_emb(nn.Module):
    def __init__(self, no_layers, dropout, embedding):
        super(ocr_Encoder_emb, self).__init__()
        self.embedding = embedding
        self.lstm = nn.LSTM(input_size = 768, hidden_size = 256, num_layers = no_layers, bidirectional = True, batch_first=True,
                             dropout =  dropout)
        self.relu = nn.ReLU(inplace = False)
        
    def forward(self, text):
        embedding = self.embedding(**text).last_hidden_state
        output, _ = self.lstm(embedding)
        return output

class entity_Encoder_emb(nn.Module):
    def __init__(self, no_layers, dropout, embedding):
        super(entity_Encoder_emb, self).__init__()
        self.embedding = embedding
        self.lstm = nn.LSTM(input_size = 768, hidden_size = 256, num_layers = no_layers, bidirectional = True, batch_first=True,
                            dropout = dropout)
        self.relu = nn.ReLU(inplace = False)
        
    def forward(self, text):

        embedding = self.embedding(**text).last_hidden_state
        _, hidden = self.lstm(embedding)
        return torch.cat((hidden[0][0:1], hidden[0][1:2]), dim=2).permute(1, 0, 2)

class Model_emb(nn.Module):
    def __init__(self, ocr_encoder, entity_encoder, no_of_classes):
        super(Model_emb, self).__init__()
        
        self.no_of_classes = no_of_classes
        self.ocr_encoder =  ocr_encoder
        self.entity_encoder = entity_encoder
        
        self.Wa = torch.nn.Parameter(torch.FloatTensor(512, 512), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.Wa)        
        self.We = torch.nn.Parameter(torch.FloatTensor(512, 512), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.We)
        
        self.Wc = torch.nn.Parameter(torch.FloatTensor(512, 512), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.Wc)
        self.Wd = torch.nn.Parameter(torch.FloatTensor(512, 512), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.Wd)
        
        self.wHc = torch.nn.Parameter(torch.FloatTensor(512, 1), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.wHc)
        self.wHd = torch.nn.Parameter(torch.FloatTensor(512, 1), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.wHd)
        
        self.fc = nn.Linear(1024, no_of_classes)
        self.relu = nn.ReLU(inplace = False)
        self.softmax1 = nn.Softmax(dim = 1)
        self.softmax2 = nn.Softmax(dim = 2)
        self.tanh = nn.Tanh()
        
        
    def forward(self, text, entity, feature):
        C = self.ocr_encoder(text)
        E = self.entity_encoder(entity)
        D = feature        
        A = (torch.matmul(torch.matmul(C, self.Wa), D.permute(0, 2, 1)) +
            torch.matmul(torch.matmul(C, self.We), E.permute(0, 2, 1)))
        
        Lc = self.softmax2(A)
        Ld = self.softmax2(A.permute(0, 2, 1))
        
        C_hat = torch.matmul(C, self.Wc).permute(0 ,2, 1)
        D_hat = torch.matmul(D, self.Wd).permute(0 ,2, 1)
        
        Hc = self.tanh(C_hat + torch.matmul(D_hat, Lc.permute(0, 2, 1)))
        Hd = self.tanh(D_hat + torch.matmul(C_hat, Ld.permute(0, 2, 1)))
        
        alpha = self.softmax2(torch.matmul(Hc.permute(0, 2, 1), self.wHc)).permute(0, 2, 1) 
        beta = self.softmax2(torch.matmul(Hd.permute(0, 2, 1), self.wHd)).permute(0, 2, 1)

        C_att = (alpha.permute(0,2,1) * C).sum(dim = 1)
        D_att = (beta.permute(0,2,1) * D).sum(dim = 1)
                  
        out = self.fc(torch.cat((C_att, D_att), dim = 1))        
        
        return out

        

vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
vgg = torch.nn.Sequential(*(list(vgg.children())[:-1]))
vgg = vgg.to(device)