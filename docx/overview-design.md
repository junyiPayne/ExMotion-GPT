# 概要设计文档 (Overview Design)

## 1. 设计理念
本系统 (ExoResearch-GPT) 专为科研实验设计，不同于工程版的黑盒模型，本系统侧重于**特征可解释性**与**实验自动化**。系统采用“双流融合架构”，旨在验证手工特征（Feature Engineering）在深度学习模型中的增益效果。

## 2. 核心架构
系统采用 **双流输入架构 (Dual-Stream Architecture)**：
* **Stream A (深度流)**: 原始波形 -> CNN -> BiLSTM -> 深层语义特征。
* **Stream B (特征流)**: 原始波形 -> FFT/RMS 计算 -> 统计特征向量。
* **Fusion (融合层)**: 深层特征 + 统计特征 -> Concat -> 分类器。


## 3. 模块化设计
* **数据层**: `DataProcessor` 负责清洗与特征计算，支持 Pandas/Numpy 接口。
* **模型层**: `CNNBiLSTMAttention` 支持通过配置动态关闭 CNN/LSTM/Attention 模块。
* **实验层**: `ExperimentRunner` 实现了自动化消融实验管线，支持一键复现论文 Table 3 结果。