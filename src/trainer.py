import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

class Trainer:
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for batch in loader:
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
        return total_loss / len(loader)

    def evaluate(self, loader):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                raw = batch['raw'].to(self.device)
                labels = batch['label'].to(self.device)
                feat = batch.get('feat')
                if feat is not None:
                    feat = feat.to(self.device)
                
                logits = self.model(raw, feat)
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        return acc, f1

    def fit(self, train_loader, val_loader, epochs=10):
        best_acc = 0.0
        print(f"   Start Training for {epochs} epochs...")
        
        for epoch in range(epochs):
            loss = self.train_epoch(train_loader)
            val_acc, val_f1 = self.evaluate(val_loader)
            
            if val_acc > best_acc:
                best_acc = val_acc
                # 科研代码通常会在这里 save model
                # torch.save(self.model.state_dict(), 'best_model.pth')
        
        return best_acc, val_f1