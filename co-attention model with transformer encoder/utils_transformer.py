import numpy as np  
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

import numpy as np  
import torch
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def score_embeddings(model,vgg, loader, device):
    model.eval()
    correct = 0
    total_size = 0
    y_hat = [] 
    y_gt = []
    with torch.no_grad():
        for(entity, text, image, role) in tqdm(loader):
          entity["input_ids"] = entity["input_ids"].to(device,dtype = torch.int32)
          entity["attention_mask"] = entity["attention_mask"].to(device,dtype = torch.int32)
          text["input_ids"] = text["input_ids"].to(device, dtype = torch.int32)
          text["attention_mask"] = text["attention_mask"].to(device, dtype = torch.int32)
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


def score(model,vgg, loader, device):
    model.eval()
    correct = 0
    total_size = 0
    y_hat = [] 
    y_gt = []

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