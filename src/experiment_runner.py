import torch
import pandas as pd
from torch.utils.data import DataLoader
from data_loader import SlidingWindowDataset
from model import CNNBiLSTMAttention
from trainer import Trainer

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
        print(">>> [ExoResearch] 启动自动化消融实验...")
        
        # 1. 准备模拟数据 (3通道, 100长度)
        # 实际应从 CSV 加载
        print(">>> 生成模拟数据集...")
        X_raw = torch.randn(100, 3, 100)
        y = torch.randint(0, 5, (100,))
        
        # 使用 DataProcessor 中的 Dataset (包含特征工程)
        dataset = SlidingWindowDataset(X_raw, y, use_features=True)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # 获取手工特征维度 (3个通道 * 5种特征 = 15)
        # 这里自动推断
        sample_feat = dataset[0]['feat']
        feat_dim = sample_feat.shape[0] * sample_feat.shape[1] 

        # 2. 遍历配置运行实验 [cite: 401-409]
        for name, cfg in self.configs.items():
            print(f"\n--> Running Experiment: {name}")
            
            # 构建模型 (传入 feature_dim 支持混合特征)
            model = ModelFactory.build(cfg, feature_dim=feat_dim)
            
            # 训练
            trainer = Trainer(model)
            acc, f1 = trainer.fit(loader)
            
            print(f"    Result -> Acc: {acc:.4f}, F1: {f1:.4f}")
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