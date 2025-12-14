import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

class Trainer:
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        
    def fit(self, loader, epochs=5): # 演示用5轮
        self.model.train()
        best_acc = 0.0
        
        for epoch in range(epochs):
            total_loss = 0
            all_preds, all_labels = [], []
            
            for batch in loader:
                # 支持字典输入 (GPT报告中的设计) [cite: 428]
                raw = batch['raw'].to(self.device)
                labels = batch['label'].to(self.device)
                feat = batch.get('feat')
                if feat is not None:
                    feat = feat.to(self.device)
                
                self.optimizer.zero_grad()
                logits = self.model(raw, feat)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
            acc = accuracy_score(all_labels, all_preds)
            if acc > best_acc:
                best_acc = acc
                
        return best_acc, f1_score(all_labels, all_preds, average='weighted')