import torch
import torch.nn as nn

class CNNBiLSTMAttention(nn.Module):
    def __init__(self, num_channels=3, num_classes=5, feature_dim=None, config=None):
        """
        支持消融配置与特征融合的模型 [cite: 205, 371]
        """
        super().__init__()
        self.cfg = config if config else {'use_cnn': True, 'use_lstm': True, 'use_attn': True}
        
        # 1. CNN 部分
        if self.cfg.get('use_cnn'):
            self.conv = nn.Sequential(
                nn.Conv1d(num_channels, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2)
            )
            lstm_input = 64
        else:
            self.conv = nn.Identity()
            lstm_input = num_channels

        # 2. LSTM 部分
        if self.cfg.get('use_lstm'):
            self.bilstm = nn.LSTM(input_size=lstm_input, hidden_size=128, 
                                  batch_first=True, bidirectional=True)
            self.dropout_lstm = nn.Dropout(0.2)
            lstm_out_dim = 256
        else:
            self.bilstm = None
            lstm_out_dim = lstm_input * 50 # 假设无池化/LSTM时的Flatten

        # 3. Attention 部分
        if self.cfg.get('use_attn'):
            self.attention = nn.Linear(lstm_out_dim, 1)

        # 4. 分类头 (支持特征融合)
        # 如果传入了手工特征(feature_dim)，则在全连接层前拼接
        fc_input = lstm_out_dim + (feature_dim if feature_dim else 0)
        
        self.classifier = nn.Sequential(
            nn.Linear(fc_input, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, raw, feat=None):
        x = raw
        
        # CNN Stream
        if self.cfg.get('use_cnn'):
            x = self.conv(x)
            # [Batch, 64, 50] -> [Batch, 50, 64]
            x = x.permute(0, 2, 1)
        elif hasattr(self.conv, 'weight'): # 如果是 Identity 则不需要 permute
             x = x.permute(0, 2, 1)

        # LSTM Stream
        if self.cfg.get('use_lstm'):
            lstm_out, _ = self.bilstm(x) # [Batch, 50, 256]
            x = self.dropout_lstm(lstm_out)
        
        # Attention Stream
        if self.cfg.get('use_attn') and self.cfg.get('use_lstm'):
            # [Batch, 50, 1]
            attn_weights = torch.softmax(torch.tanh(self.attention(x)), dim=1)
            # Context Vector [Batch, 256]
            context = torch.sum(attn_weights * x, dim=1)
        else:
            # 如果没有 Attention 或 LSTM，简单的 Global Average Pooling 或 Flatten
            if len(x.shape) == 3:
                context = x.mean(dim=1) 
            else:
                context = x

        # 特征融合 (Hybrid Fusion)
        if feat is not None:
            # Flatten feat if needed
            feat = feat.flatten(start_dim=1)
            context = torch.cat([context, feat], dim=1)
            
        return self.classifier(context)