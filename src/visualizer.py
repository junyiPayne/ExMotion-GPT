import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import numpy as np

def plot_confusion_matrix(y_true, y_pred, classes=None, save_path="confusion_matrix.png"):
    """
    绘制并保存混淆矩阵 [cite: 371]
    """
    if classes is None:
        classes = ['RS', 'MA', 'RM', 'DLB', 'SLB']
        
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    print(f"   >>> Confusion Matrix saved to {save_path}")
    plt.close()

def plot_tsne(features, labels, save_path="tsne_plot.png"):
    """
    绘制 t-SNE 特征分布图 [cite: 369]
    Args:
        features: 模型倒数第二层的输出向量 (Batch, Hidden)
        labels: 真实标签
    """
    print("   >>> Computing t-SNE (this may take a while)...")
    tsne = TSNE(n_components=2, random_state=42)
    embedding = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Feature Space')
    plt.savefig(save_path)
    print(f"   >>> t-SNE plot saved to {save_path}")
    plt.close()