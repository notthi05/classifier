import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import numpy as np
from sklearn.metrics import roc_auc_score

# テストデータの読み込み
test_df = pd.read_csv('test_filename.csv', encoding='UTF-8', sep=';')

# データの前処理（小文字化）
test_df['text'] = test_df['text'].str.lower()

# 入力データとラベルの抽出
test_texts = test_df['text'].values
test_labels = test_df['label'].values

# BERTのトークナイザーの準備
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# データのトークン化とエンコーディング
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=128)

# 入力データと注意マスクのテンソル化
test_inputs = torch.tensor(test_encodings['input_ids'])
test_masks = torch.tensor(test_encodings['attention_mask'])
test_labels = torch.tensor(test_labels)

# データセットの作成
test_dataset = TensorDataset(test_inputs, test_masks, test_labels)

# バッチサイズの設定
batch_size = 32

# データローダーの作成
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# BERTモデルの準備
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# GPUが利用可能な場合は、モデルをGPUに移動
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 損失関数と最適化手法の設定
criterion = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=3e-5)

# 最大エポック数の設定
num_epochs = 10

# 保存したモデルの読み込み
model.load_state_dict(torch.load('model_x.pth'))

# テストデータでの予測と評価
model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs, masks, labels = batch

        outputs = model(inputs, attention_mask=masks)
        logits = outputs.logits

        # ソフトマックス関数を使用して確率に変換
        probabilities = torch.softmax(logits, dim=1)

        predicted = torch.argmax(probabilities, dim=1).cpu().numpy()

        all_predictions.extend(probabilities.cpu().numpy())  # 確率を追加
        all_labels.extend(labels.cpu().numpy())

predicted_labels = np.argmax(all_predictions, axis=1)

# 予測結果を含むデータフレームを作成
results_df = pd.DataFrame({'True_Label': all_labels,
                            'Predicted_Label': predicted_labels,
                            'Text': test_texts
                            })

# CSVファイルに保存
results_df.to_csv('results.csv', index=False)

# 混同行列
confusion = confusion_matrix(all_labels, predicted_labels)
print("Confusion Matrix:")
print(confusion)

# Precision, Recall, F1-score
report = classification_report(all_labels, predicted_labels, target_names=['Class 0', 'Class 1', 'Class 2'])
print("Classification Report:")
print(report)

class_auc_scores = []
for class_index in range(3):
    # クラスごとの真のラベルと確率を抽出
    class_true_labels = [1 if label == class_index else 0 for label in all_labels]
    class_probabilities = [probability[class_index] for probability in all_predictions]

    # クラスごとのAUCを計算
    class_auc = roc_auc_score(class_true_labels, class_probabilities)
    class_auc_scores.append(class_auc)
    print(f"Class {class_index} AUC: {class_auc:.4f}")

# 平均AUC（マクロ平均）を計算
macro_auc = np.mean(class_auc_scores)
print(f"Macro-average AUC: {macro_auc:.4f}")
