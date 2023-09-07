import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import numpy as np
from sklearn.metrics import roc_auc_score

# データの読み込み
df = pd.read_csv('Reuro.csv', encoding='UTF-8', sep=';')

# テキストデータを小文字化
df['text'] = df['text'].str.lower()

# 入力データとラベルの抽出
texts = df['text'].values
labels = df['label'].values

# データの分割
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.3, random_state=42)

# BERTのトークナイザーの準備
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# データのトークン化とエンコーディング
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=128)

# 入力データと注意マスクのテンソル化
train_inputs = torch.tensor(train_encodings['input_ids'])
train_masks = torch.tensor(train_encodings['attention_mask'])
train_labels = torch.tensor(train_labels)
test_inputs = torch.tensor(test_encodings['input_ids'])
test_masks = torch.tensor(test_encodings['attention_mask'])
test_labels = torch.tensor(test_labels)

# データセットの作成
train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
test_dataset = TensorDataset(test_inputs, test_masks, test_labels)

# バッチサイズの設定
batch_size = 32

# データローダーの作成
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
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

# 学習ループ
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs, masks, labels = batch

        optimizer.zero_grad()
        outputs = model(inputs, attention_mask=masks, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f'Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}')

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

# 混同行列
confusion = confusion_matrix(all_labels, predicted_labels)
print("Confusion Matrix:")
print(confusion)

# Precision, Recall, F1-score
report = classification_report(all_labels, predicted_labels, target_names=['Class 0', 'Class 1', 'Class 2'])
print("Classification Report:")
print(report)

# AUC
all_labels = np.array(all_labels).reshape(-1, 1)
all_predictions = np.array(all_predictions)
auc = roc_auc_score(all_labels, all_predictions, multi_class='ovr', average='macro')
print(f"AUC: {auc:.4f}")
