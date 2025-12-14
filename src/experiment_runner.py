import torch
import pandas as pd
from torch.utils.data import DataLoader
from data_loader import SlidingWindowDataset
from model import CNNBiLSTMAttention
from trainer import Trainer
from torch.utils.data import random_split

# 简单的模型工厂
class ModelFactory:
    @staticmethod
    def build(config, feature_dim=None):
        return CNNBiLSTMAttention(config=config, feature_dim=feature_dim)

class ExperimentRunner:
    def __init__(self):
        self.results = {}
        # 定义配置池 (对应报告 Table 3) [cite: 396-399]
        self.configs = {
            "CNN_Only": {"use_cnn": True, "use_lstm": False, "use_attn": False},
            "BiLSTM_Only": {"use_cnn": False, "use_lstm": True, "use_attn": False},
            "Proposed_Full": {"use_cnn": True, "use_lstm": True, "use_attn": True}
        }

    def run_benchmark(self):
        print(">>> [ExoResearch] 启动自动化消融实验 (Automated Ablation Study)...")
        
        # 1. 准备数据
        print(">>> 生成模拟数据集 (含混合特征)...")
        # 模拟 200 个样本 (比之前多一点，方便划分)
        X_raw = torch.randn(200, 3, 100)
        y = torch.randint(0, 5, (200,))
        
        full_dataset = SlidingWindowDataset(X_raw, y, use_features=True)
        
        # --- 关键修改：科研严谨性，划分训练集和验证集 (80/20) ---
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # 自动推断特征维度
        sample_feat = full_dataset[0]['feat']
        feat_dim = sample_feat.numel() # 3通道 * 5特征 = 15

        # 2. 遍历配置运行实验
        for name, cfg in self.configs.items():
            print(f"\n--> Running Experiment: {name}")
            
            model = ModelFactory.build(cfg, feature_dim=feat_dim)
            trainer = Trainer(model)
            
            # 传入 train 和 val loader
            acc, f1 = trainer.fit(train_loader, val_loader, epochs=5)
            
            print(f"    [Final Result] Val Acc: {acc:.4f}, Val F1: {f1:.4f}")
            self.results[name] = {"Accuracy": acc, "F1": f1}
            
        self.export_report()

    def export_report(self):
        # 生成对比报表 [cite: 410-415]
        df = pd.DataFrame(self.results).T
        print("\n>>> 实验结果汇总 (Ablation Study Report):")
        print(df)

if __name__ == "__main__":
    runner = ExperimentRunner()
    runner.run_benchmark()