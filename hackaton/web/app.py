from flask import Flask, render_template, request, send_file, Response
import pandas as pd
import json
import os
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from nn_model import Net
from osint.osint_checker import check_osint

MODEL_PATH = os.path.join('model', 'nn_model.pth')
FEATURES_PATH = os.path.join('model', 'features.json')
SCALER_PATH = os.path.join('model', 'scaler.json')

app = Flask(__name__)

def load_features():
    with open(FEATURES_PATH) as f:
        return json.load(f)

def load_scaler():
    with open(SCALER_PATH) as f:
        values = json.load(f)
    n = len(values) // 2
    scaler = StandardScaler()
    scaler.mean_ = np.array(values[:n])
    scaler.scale_ = np.array(values[n:])
    return scaler

def extract_features(df, feature_cols):
    cons_df = pd.json_normalize(df['consumption']).fillna(0)
    month_mapping = {str(i): f"month_{i-1}" for i in range(1, 13)}
    cons_df = cons_df.rename(columns=month_mapping)
    for i in range(12):
        col = f"month_{i}"
        if col not in cons_df.columns:
            cons_df[col] = 0
    cons_df = cons_df[[f"month_{i}" for i in range(12)]]

    df = pd.concat([df.drop('consumption', axis=1), cons_df], axis=1)
    for col in ['totalArea', 'roomsCount', 'residentsCount']:
        df[col] = df[col].fillna(df[col].median() if col in df else 0)

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
        .str.extract(r'(\d+)')[0]
        .astype(float)
        .fillna(-1)
    )

    X = df[feature_cols].fillna(0)
    return df, X

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']
    df = pd.read_json(file)
    feature_cols = load_features()
    scaler = load_scaler()
    df_proc, X = extract_features(df.copy(), feature_cols)
    X_scaled = scaler.transform(X)

    model = Net(X_scaled.shape[1])
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_tensor).squeeze()
        probs = torch.sigmoid(logits).numpy()

    def generate():
        results = []
        for i, row in df_proc.iterrows():
            osint_flag, osint_words, osint_sources, osint_links = check_osint(row['address'])
            risk = float(probs[i])
            reasons = []
            if risk > 0.7:
                reasons.append('Высокое потребление')
            if osint_flag:
                reasons.append('Найден бизнес по адресу')
            result = {
                'accountId': int(row['accountId']),
                'address': row['address'],
                'risk_probability': risk,
                'osint_flag': int(osint_flag),
                'osint_words': ";".join(osint_words),
                'osint_sources': ";".join(osint_sources),
                'osint_links': ";".join(osint_links),
                'reasons': ', '.join(reasons) if reasons else "Нет явных признаков"
            }
            results.append(result)
            yield json.dumps(result, ensure_ascii=False) + '\n'
        # Save all to file after stream
        with open('results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    return Response(generate(), mimetype='text/plain')

@app.route('/download')
def download():
    # Преобразуем результаты в xlsx перед отправкой
    with open('results.json', encoding='utf-8') as f:
        results = json.load(f)
    df = pd.DataFrame(results)
    # Приводим к нужным колонкам и форматам
    df = df.rename(columns={
        'accountId': 'accountId',
        'address': 'Адрес',
        'risk_probability': 'Риск (%)',
        'osint_flag': 'Бизнес-флаг',
        'reasons': 'Причины'
    })
    df['Риск (%)'] = (df['Риск (%)'] * 100).round(2)
    df = df[['accountId', 'Адрес', 'Риск (%)', 'Бизнес-флаг', 'Причины']]

    xlsx_path = 'results.xlsx'
    df.to_excel(xlsx_path, index=False)
    return send_file(xlsx_path, as_attachment=True, download_name="results.xlsx")

@app.route('/search.html')
def search():
    return render_template('search.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
