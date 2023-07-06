import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

df = pd.read_csv('data.csv')

# データの前処理（小文字化）
df['text'] = df['text'].str.lower()

texts = df['text'].values
labels = df['label'].values

# データの分割
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.3, random_state=42)

# BERTのトークナイザーの準備
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# BERTに入力する形式にデータを変換
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True, max_length=128)

# データセットの作成
train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_encodings['input_ids']),
                                               torch.tensor(train_encodings['attention_mask']),
                                               torch.tensor(train_labels))
test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_encodings['input_ids']),
                                              torch.tensor(test_encodings['attention_mask']),
                                              torch.tensor(test_labels))

# バッチサイズの設定
batch_size = 32

# データローダーの作成
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# BERTモデルの準備
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# GPUが利用可能な場合は、モデルをGPUに移動
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 最適化手法と学習率の設定
optimizer = AdamW(model.parameters(), lr=3e-5)

# 最大エポック数の設定
num_epochs = 10

# 学習ループ
for epoch in range(num_epochs):
    model.train()

    for batch in train_dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs, masks, labels = batch

        optimizer.zero_grad()

        outputs = model(inputs, attention_mask=masks, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # テストデータでの評価
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs, masks, labels = batch

            outputs = model(inputs, attention_mask=masks)
            logits = outputs.logits
            _, predicted = torch.max(logits, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Epoch {epoch+1}/{num_epochs} - Accuracy: {accuracy:.4f}')