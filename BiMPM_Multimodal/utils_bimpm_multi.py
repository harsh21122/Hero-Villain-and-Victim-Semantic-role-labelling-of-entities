from google.colab import drive
import shutil
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import torch
import numpy as np

def save_model(model):
  #Saving the best model to drive
  drive.mount('/content/drive')
  shutil.copy("/content/" + model, "/content/drive/MyDrive/Dl thesis/model")
  print("Model Saved")
  drive.flush_and_unmount()

def load_model(model):
  drive.mount('/content/drive')
  shutil.copy("/content/drive/MyDrive/Dl thesis/model/" + model, '/content/')
  print("Model Loaded")
  drive.flush_and_unmount()

def score(model, encoder, loader, device):
    model.eval()
    encoder.eval()
    correct = 0
    total_size = 0
    y_hat = [] 
    y_gt = []

    for batch, (text, image, entity, role) in enumerate(loader):
        text = text.to(device=device, dtype = torch.int32)
        entity = entity.to(device=device, dtype = torch.int32)
        role = role.to(device=device, dtype = torch.int64)
        with torch.no_grad():
          imgs_f, img_mean, img_att = encoder(image.to(device = device, dtype = torch.float32)) 
        outputs,_,_= model( text, entity, imgs_f, img_mean, img_att)  
        
        pred = torch.argmax(outputs, dim = 1)
        correct += (role == pred).sum()
        total_size += outputs.shape[0]
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