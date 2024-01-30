import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# 教師データの読み込み
train_df = pd.read_csv('name.csv', encoding='UTF-8', sep=';')

# データの前処理（小文字化）
train_df['text'] = train_df['text'].str.lower()

# 入力データとラベルの抽出
train_texts = train_df['text'].values
train_labels = train_df['label'].values

# BERTのトークナイザーの準備
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# データのトークン化とエンコーディング
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)

# 入力データと注意マスクのテンソル化
train_inputs = torch.tensor(train_encodings['input_ids'])
train_masks = torch.tensor(train_encodings['attention_mask'])
train_labels = torch.tensor(train_labels)

# データセットの作成
train_dataset = TensorDataset(train_inputs, train_masks, train_labels)

# バッチサイズの設定
batch_size = 32

# データローダーの作成
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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

# モデルとオプティマイザの保存
torch.save(model.state_dict(), 'model_x.pth')
torch.save(optimizer.state_dict(), 'opt_x.pth')
