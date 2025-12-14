# ExoResearch-GPT: 上肢外骨骼运动意图识别研究平台

本项目基于 **混合特征工程 (Hybrid Feature Engineering)** 与 **CNN-BiLSTM-Attention** 架构，旨在探索不同模型组件对 sEMG 信号分类性能的影响。

## 1. 核心特性
* **混合特征**: 融合 Deep Learning 特征与 RMS/MFCC 等手工特征。
* **自动化实验**: 内置 `ExperimentRunner`，一键运行 Baseline A/B/C 与 Proposed 模型的消融对比。
* **严谨协议**: 遵循 LOOCV (留一法) 验证标准。

## 2. 快速开始
### 安装
```bash
pip install -r requirements.txt
