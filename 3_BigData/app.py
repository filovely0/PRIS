import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    StackingRegressor
)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

MAX_SAMPLE_ROWS = 100_000_000
CHUNK_SIZE = 500_000


def sample_dataframe(path, usecols):
    samples = []
    total = 0
    for chunk in pd.read_csv(
            path,
            usecols=usecols,
            chunksize=CHUNK_SIZE,
            low_memory=False,
            on_bad_lines='skip'
    ):
        remaining = MAX_SAMPLE_ROWS - total
        if remaining <= 0:
            break
        frac = min(remaining / len(chunk), 1.0)
        samples.append(chunk.sample(frac=frac, random_state=42))
        total += int(len(chunk) * frac)
    if samples:
        return pd.concat(samples, ignore_index=True)
    return pd.read_csv(path, usecols=usecols, low_memory=False, on_bad_lines='skip')


def preprocess(df, x_col, y_col):
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'], format='%Y-%m-%d')
    df['flight_dow'] = df['FL_DATE'].dt.dayofweek
    df['sin_dow'] = np.sin(2 * np.pi * df['flight_dow'] / 7)
    df['cos_dow'] = np.cos(2 * np.pi * df['flight_dow'] / 7)
    df['flight_month'] = df['FL_DATE'].dt.month
    df['sin_mon'] = np.sin(2 * np.pi * df['flight_month'] / 12)
    df['cos_mon'] = np.cos(2 * np.pi * df['flight_month'] / 12)

    for col in ('CRS_DEP_TIME', 'DEP_TIME'):
        df[col] = df[col].fillna(0).astype(int)
        df[f'{col}_mins'] = (df[col] // 100) * 60 + (df[col] % 100)

    df['hour'] = df['CRS_DEP_TIME_mins'] // 60
    df['sin_h'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_h'] = np.cos(2 * np.pi * df['hour'] / 24)

    df['ORIGIN_DEST'] = df['ORIGIN'] + '_' + df['DEST']

    df['carrier_delay_mean'] = df.groupby('OP_CARRIER')['DEP_DELAY'].transform('mean')
    df['route_delay_mean'] = df.groupby('ORIGIN_DEST')['DEP_DELAY'].transform('mean')

    features = [
        'sin_dow', 'cos_dow', 'sin_mon', 'cos_mon',
        'CRS_DEP_TIME_mins', 'sin_h', 'cos_h',
        'TAXI_OUT', 'WHEELS_OFF',
        'carrier_delay_mean', 'route_delay_mean',
        'OP_CARRIER', 'ORIGIN', 'DEST', 'ORIGIN_DEST'
    ]
    for c in (x_col, y_col):
        if c in df.columns and c not in features:
            features.append(c)

    df = df.dropna(subset=features + [y_col])
    low, high = df[y_col].quantile([0.01, 0.99])
    df = df[df[y_col].between(low, high)]

    return df, features


@app.route('/')
def home():
    return render_template('upload.html')


@app.route('/preview', methods=['POST'])
def preview():
    f = request.files['file']
    path = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(path)

    df_preview = pd.read_csv(
        path,
        usecols=[
            'FL_DATE', 'OP_CARRIER', 'ORIGIN', 'DEST',
            'CRS_DEP_TIME', 'DEP_TIME',
            'DEP_DELAY', 'TAXI_OUT', 'WHEELS_OFF'
        ],
        nrows=1000,
        on_bad_lines='skip'
    )
    df_preview.to_csv(os.path.join(UPLOAD_FOLDER, 'last_upload.csv'), index=False)

    numeric_cols = df_preview.select_dtypes(include='number').columns.tolist()
    preview_html = df_preview.head().to_html(classes='table table-bordered', index=False)
    return render_template('choose_columns.html', columns=numeric_cols, preview_table=preview_html)


@app.route('/train', methods=['POST'])
def train():
    csv_path = os.path.join(UPLOAD_FOLDER, 'last_upload.csv')
    x_col = request.form['x_column']
    y_col = request.form['y_column']

    df_sample = sample_dataframe(
        csv_path,
        usecols=[
            'FL_DATE', 'OP_CARRIER', 'ORIGIN', 'DEST',
            'CRS_DEP_TIME', 'DEP_TIME',
            'DEP_DELAY', 'TAXI_OUT', 'WHEELS_OFF'
        ]
    )

    df, features = preprocess(df_sample, x_col, y_col)
    X = df[features]
    y = df[y_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    cat_feats = ['OP_CARRIER', 'ORIGIN', 'DEST', 'ORIGIN_DEST']
    preproc = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_feats)
    ], remainder='passthrough')

    estimators = [
        ('hgb', HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=10, random_state=42
        )),
        ('rf', RandomForestRegressor(
            n_estimators=200, max_depth=12, n_jobs=-1, random_state=42
        )),
        ('et', ExtraTreesRegressor(
            n_estimators=200, max_depth=12, n_jobs=-1, random_state=42
        ))
    ]
    final_est = HistGradientBoostingRegressor(
        max_iter=150, learning_rate=0.05, max_depth=6, random_state=42
    )
    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=final_est,
        cv=3,
        n_jobs=-1,
        passthrough=True
    )

    model = Pipeline([
        ('prep', preproc),
        ('stack', stack)
    ])

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, preds, alpha=0.3, label='Прогнозы')
    mn, mx = min(y_test.min(), preds.min()), max(y_test.max(), preds.max())
    plt.plot([mn, mx], [mn, mx], '--r', linewidth=2, label='y = x')
    plt.xlabel('Фактическая задержка (мин)')
    plt.ylabel('Предсказанная задержка (мин)')
    plt.title(f'Задержка отправления (MSE = {mse:.2f})')
    plt.legend(loc='upper left')
    plt.grid(True)
    plot_path = os.path.join(STATIC_FOLDER, 'plot.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    return render_template('result.html', mse=round(mse, 2), plot_url='static/plot.png')


if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
