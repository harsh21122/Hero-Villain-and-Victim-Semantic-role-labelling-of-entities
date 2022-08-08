import numpy as np  
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np  
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def score(model, loader, device):
    model.eval()
    correct = 0
    total_size = 0
    y_hat = [] 
    y_gt = []
    with torch.no_grad():
        for (text, entity, role) in tqdm(loader):
            text = text.to(device=device, dtype = torch.int32)
            entity = entity.to(device=device, dtype = torch.int32)
            role = role.to(device=device, dtype = torch.int64)
            hidden = [torch.zeros((2, text.shape[0], 128), dtype = torch.float32).to(device=device),
                      torch.zeros((2, text.shape[0], 128), dtype = torch.float32).to(device=device)]        
            y_pred = model(text, entity, hidden)  
            
            pred = torch.argmax(y_pred, dim = 1)
            correct += (role == pred).sum()
            total_size += y_pred.shape[0]
            y_gt.extend(role.cpu().int().numpy())
            y_hat.extend(pred.cpu().int().numpy())

    macro = f1_score(y_gt, y_hat, average='macro') 
    cf_matrix = confusion_matrix(y_gt, y_hat)
    sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cbar = False) 
    plt.show()
    return (correct.item() * 100) / total_size, macro


def embedding(PATH = r'.\glove.6B.100d.txt'):
    
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