import pandas as pd
import json
import os
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from joblib import dump
from nn_model import Net
import torch.nn.functional as F

def load_dataset(path):
    df = pd.read_json(path)
    cons_df = pd.json_normalize(df['consumption']).fillna(0)

    # Преобразуем ключи: "1"–"12" → "month_0"–"month_11"
    month_mapping = {str(i): f"month_{i-1}" for i in range(1, 13)}
    cons_df = cons_df.rename(columns=month_mapping)
    for i in range(12):
        col = f"month_{i}"
        if col not in cons_df.columns:
            cons_df[col] = 0
    cons_df = cons_df[[f"month_{i}" for i in range(12)]]

    df = pd.concat([df.drop('consumption', axis=1), cons_df], axis=1)

    for col in ['totalArea', 'roomsCount', 'residentsCount']:
        df[col] = df[col].fillna(df[col].median())

    df['cons_per_res'] = df[[f'month_{i}' for i in range(12)]].sum(axis=1) / (df['residentsCount'] + 0.1)
    df['summer_use'] = df[['month_5', 'month_6', 'month_7']].sum(axis=1)
    df['winter_use'] = df[['month_11', 'month_0', 'month_1']].sum(axis=1)
    df['season_diff'] = df['summer_use'] - df['winter_use']
    df['season_ratio'] = df['summer_use'] / (df['winter_use'] + 1)
    df['peak_month'] = df[[f'month_{i}' for i in range(12)]].max(axis=1)
    df['avg_month'] = df[[f'month_{i}' for i in range(12)]].mean(axis=1)
    df['peak_to_avg_ratio'] = df['peak_month'] / (df['avg_month'] + 1)
    df['winter_zero_months'] = df[['month_11', 'month_0', 'month_1']].apply(lambda row: (row == 0).sum(), axis=1)
    df['summer_peak_month'] = (
    df[['month_5', 'month_6', 'month_7']]
    .idxmax(axis=1)
    .str.extract(r'(\\d+)')[0]
    .astype(float)
    .fillna(-1)  # или оставь NaN, если хочешь
)


    df = df[df['isCommercial'].notnull()].copy()
    df['isCommercial'] = df['isCommercial'].astype(int)

    features = [f'month_{i}' for i in range(12)] + [
        'totalArea', 'roomsCount', 'residentsCount', 'cons_per_res',
        'season_diff', 'season_ratio', 'peak_month', 'avg_month',
        'peak_to_avg_ratio', 'winter_zero_months', 'summer_peak_month'
    ]
    return df[features], df['isCommercial'], features

def train_nn_model(X, y, features):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y.values, test_size=0.2, random_state=42, stratify=y)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)

    model = Net(X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]))

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            val_output = model(X_val_tensor)
            val_loss = criterion(val_output, y_val_tensor).item()
            print(f"NN Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

    os.makedirs('web/model', exist_ok=True)
    torch.save(model.state_dict(), 'web/model/nn_model.pth')
    with open('web/model/scaler.json', 'w') as f:
        json.dump(scaler.mean_.tolist() + scaler.scale_.tolist(), f)
    with open('web/model/features.json', 'w') as f:
        json.dump(features, f)
    print("Нейросеть обучена и сохранена.")

def train_lgb_model(X, y):
    X = X.copy()
    X = X.astype(float)  # преобразуем всё в float
    
    print("🧠 LightGBM training on:", X.shape, "features")
    
    model = LGBMClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight='balanced',
        learning_rate=0.05
    )
    
    model.fit(X, y)
    
    # Проверим важность признаков
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({'feature': X.columns, 'importance': importances})
    print("🔍 Feature importances:")
    print(feat_imp.sort_values(by='importance', ascending=False))

    dump(model, 'web/model/ml_model.joblib')
    print("✅ LightGBM обучен и сохранён.")


if __name__ == '__main__':
    X_raw, y, features = load_dataset('web/dataset_train.json')  # не скейлим для LightGBM
    train_nn_model(X_raw, y, features)  # внутри будет скейлинг для нейросети
    train_lgb_model(X_raw, y)           # подаём сырые данные
