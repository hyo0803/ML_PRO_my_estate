"""Обучение модели и сохранение joblib-бандла.

Из ноутбука:
    from estate.ml.train import train, check
    bundle = train()

Из терминала:
    python -m estate.ml.train
"""
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import sklearn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from estate.config import settings
from estate.ml.preprocess import (
    ENGINEERED_FEATURES,
    RAW_FEATURES,
    TARGET,
    FeatureEngineer,
    load_data,
)

MODEL_NAME = 'estate-rf'
MODEL_VERSION = '0.1.0'
RANDOM_STATE = 42
TEST_SIZE = 0.2

# MLflow: имя в реестре и гейт. Гейт по test MAPE (в процентах, меньше — лучше):
# новая версия становится champion, только если MAPE ниже на MIN_GAIN п.п.
REGISTRY_NAME = os.getenv('MODEL_NAME', 'estate')
EXPERIMENT = os.getenv('MLFLOW_EXPERIMENT', 'estate')
MIN_GAIN = float(os.getenv('GATE_MIN_GAIN', '0.2'))

# Найдено RandomizedSearchCV в research/baseline.ipynb (CV R2 = 0.8957)
RF_PARAMS: dict[str, Any] = dict(
    n_estimators=60,
    max_depth=int(os.getenv('MAX_DEPTH', '10')),
    min_samples_leaf=20,
    max_features=0.5,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

# Доля самых недооценённых объектов, которую отсекает порог
UNDERPRICED_QUANTILE = 0.10


def build_pipeline():
    """Препроцессинг и модель одним Pipeline."""
    return Pipeline([
        ('features', FeatureEngineer(random_state=RANDOM_STATE)),
        ('scaler', RobustScaler()),
        ('model', RandomForestRegressor(**RF_PARAMS)),
    ])


def evaluate(pipeline, X, y):
    """R2 считается в лог-шкале, RMSE и MAPE — в рублях."""
    pred_log = pipeline.predict(X)
    true_rub, pred_rub = np.expm1(y), np.expm1(pred_log)
    return {
        'r2': float(r2_score(y, pred_log)),
        'rmse_rub': float(np.sqrt(mean_squared_error(true_rub, pred_rub))),
        'mape_pct': float(mean_absolute_percentage_error(true_rub, pred_rub) * 100),
    }


def compute_threshold(pipeline, X_train, y_train):
    """Порог классификации «объект недооценён».

    Остаток = факт - прогноз в лог-шкале. Сильно отрицательный остаток
    означает, что объект продаётся дешевле, чем предсказывает модель.
    """
    residuals = y_train - pipeline.predict(X_train)
    return float(np.quantile(residuals, UNDERPRICED_QUANTILE))


def train(data_path=None, model_path=None):
    """Загрузка -> обучение -> метрики -> joblib-бандл."""
    data_path = data_path or settings.data_path
    model_path = model_path or settings.model_path

    print(f'Загрузка {data_path}...')
    df = load_data(data_path)
    print(f'  записей после очистки: {len(df):,}')

    X = df[RAW_FEATURES]
    y = np.log1p(df[TARGET].values)   # обучаемся на логарифме цены

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f'  train: {len(X_train):,} | test: {len(X_test):,}')

    print('Обучение пайплайна...')
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    metrics = {
        'train': evaluate(pipeline, X_train, y_train),
        'test': evaluate(pipeline, X_test, y_test),
    }
    print(f"  test R2={metrics['test']['r2']:.4f} | "
          f"RMSE={metrics['test']['rmse_rub']/1e6:.3f} млн руб. | "
          f"MAPE={metrics['test']['mape_pct']:.2f}%")

    threshold = compute_threshold(pipeline, X_train, y_train)
    print(f'  порог: {threshold:.4f}')

    metadata = {
        'model_name': MODEL_NAME,
        'model_version': MODEL_VERSION,
        'trained_at': datetime.now(UTC).isoformat(timespec='seconds'),
        'n_train': int(len(X_train)),
        # Контракт сервиса: поля запроса в нужном порядке
        'features': list(RAW_FEATURES),
        'engineered_features': list(ENGINEERED_FEATURES),
        'target': TARGET,
        'target_transform': 'log1p',
        'inverse_transform': 'expm1',
        'threshold': threshold,
        'threshold_meaning': (
            'residual = log1p(факт) - predict(X); residual < threshold -> '
            'объект недооценён'
        ),
        'metrics': metrics,
        'model_params': RF_PARAMS,
        'sklearn_version': sklearn.__version__,
    }

    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {'pipeline': pipeline, 'metadata': metadata}
    joblib.dump(bundle, path, compress=3)
    print(f'Бандл сохранён: {path} ({path.stat().st_size / 1e6:.1f} МБ)')

    if os.getenv('MLFLOW_TRACKING_URI'):
        log_to_mlflow(pipeline, metadata, Path(data_path))

    return bundle


def feature_importance_figure(pipeline):
    """Свой артефакт прогона: важности признаков случайного леса."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    importances = pipeline.named_steps['model'].feature_importances_
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(np.array(ENGINEERED_FEATURES)[order], importances[order])
    ax.set_title('RandomForest feature importances')
    fig.tight_layout()
    return fig


def champion_mape(client):
    try:
        mv = client.get_model_version_by_alias(REGISTRY_NAME, 'champion')
    except MlflowException:
        return None, None
    return mv.version, client.get_run(mv.run_id).data.metrics.get('test_mape_pct')


def log_to_mlflow(pipeline, metadata, data_path):
    """Прогон в MLflow, регистрация версии и гейт по MAPE."""
    import skops.io as sio

    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()
    test = metadata['metrics']['test']
    trusted = sio.get_untrusted_types(data=sio.dumps(pipeline))

    with mlflow.start_run() as run:
        mlflow.log_params({**RF_PARAMS, 'region': settings.region,
                           'data': str(data_path),
                           'data_md5': hashlib.md5(data_path.read_bytes()).hexdigest()})
        mlflow.log_metrics({f'{split}_{k}': float(v)
                            for split, m in metadata['metrics'].items() for k, v in m.items()})
        mlflow.log_metric('threshold', metadata['threshold'])
        mlflow.log_dict(metadata, 'metadata.json')
        mlflow.log_figure(feature_importance_figure(pipeline), 'feature_importances.png')
        info = mlflow.sklearn.log_model(pipeline, name='model', registered_model_name=REGISTRY_NAME,
                                        skops_trusted_types=trusted)
        version = info.registered_model_version

    old_version, old_mape = champion_mape(client)
    mape = float(test['mape_pct'])
    promoted = bool(old_mape is None or mape < old_mape - MIN_GAIN)
    client.set_registered_model_alias(REGISTRY_NAME, 'challenger', version)
    if promoted:
        client.set_registered_model_alias(REGISTRY_NAME, 'champion', version)

    result = {'run_id': run.info.run_id, 'version': version, 'max_depth': RF_PARAMS['max_depth'],
              'test_mape_pct': round(mape, 3), 'champion_before': old_version,
              'champion_mape_before': old_mape, 'min_gain': MIN_GAIN, 'promoted': promoted}
    print(json.dumps(result, ensure_ascii=False))
    return result


def check(model_path=None, sample=None):
    """Контроль: бандл читается и предсказывает так же, как сервис."""
    bundle = joblib.load(model_path or settings.model_path)
    pipeline, meta = bundle['pipeline'], bundle['metadata']

    if sample is None:
        sample = load_data().head(5)
    frame = sample.reindex(columns=meta['features'])   # как в app.py сервиса

    price = np.expm1(pipeline.predict(frame))
    print(f"{meta['model_name']} {meta['model_version']}, порог {meta['threshold']:.4f}")
    print('Прогноз:', ', '.join(f'{p/1e6:.2f} млн' for p in price))
    return price


if __name__ == '__main__':
    train()
    check()
