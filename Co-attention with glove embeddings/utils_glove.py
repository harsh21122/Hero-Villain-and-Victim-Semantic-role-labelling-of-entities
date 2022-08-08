import numpy as np  
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

import numpy as np  
import torch
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def score(model,vgg, loader, device):
    model.eval()
    correct = 0
    total_size = 0
    y_hat = [] 
    y_gt = []
    with torch.no_grad(): 
        for batch, (text, image, entity, role) in enumerate(loader):
            text = text.to(device=device, dtype = torch.int32)
            entity = entity.to(device=device, dtype = torch.int32)
            role = role.to(device=device, dtype = torch.int64)
            feature = vgg(image.to(device = device, dtype = torch.float32))
            feature = torch.flatten(feature, start_dim=2)
            feature = feature.permute(0, 2, 1)
            y_pred = model(text, entity, feature)   
            
            pred = torch.argmax(y_pred, dim = 1)
            correct += (role == pred).sum()
            total_size += y_pred.shape[0]
            y_gt.extend(role.cpu().int().numpy())
            y_hat.extend(pred.cpu().int().numpy())
    
        macro = f1_score(y_gt, y_hat, average='macro') 
        weighted = f1_score(y_gt, y_hat, average='weighted')
        print(confusion_matrix(y_gt, y_hat))

    return (correct.item() * 100) / total_size, macro, weighted


def embedding(PATH = r'C:\Users\fahad\Glove\glove.6B.100d.txt'):
    
    vocab, embeddings = [], []
    file = open(PATH, encoding="utf8")
    content = file.read().strip().split('\n')

    for i in range(len(content)):
        i_word = content[i].split(' ')[0]
        i_embeddings = [float(val) for val in content[i].split(' ')[1:]]
        vocab.append(i_word)
        embeddings.append(i_embeddings)
        
    vocab.insert(0, '<PAD>')
    vocab.insert(1, '<UNK>')
        
    vocab_np = np.array(vocab)
    embeddings_np = np.array(embeddings)

    # Padding and Unknowns are all zeros and average of all vectors respectively
    pad_emb = np.zeros((1, embeddings_np.shape[1]))   
    unk_emb = np.mean(embeddings_np, axis=0, keepdims=True)   

    #insert embeddings for pad and unk tokens at top of embeddings_np
    embeddings_np = np.vstack((pad_emb, unk_emb, embeddings_np))
    
    return vocab, embeddings_np