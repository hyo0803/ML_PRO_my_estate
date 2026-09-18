"""Препроцессинг: очистка данных и генерация признаков.

Один и тот же код работает для train и test — вся логика собрана в
трансформере FeatureEngineer, который входит в Pipeline. Статистики
(медианы, границы клиппинга, центры кластеров) обучаются в fit на train
и применяются в transform без пересчёта.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

from estate.config import settings

MOSCOW_CENTER = (55.7558, 37.6173)

TARGET = 'price'

# Контракт сервиса: поля, которые присылает клиент. Это же значение
# уходит в metadata['features'] — по нему сервис выстраивает колонки.
RAW_FEATURES = [
    'date', 'geo_lat', 'geo_lon', 'building_type',
    'level', 'levels', 'rooms', 'area', 'kitchen_area', 'object_type',
]

# Признаки, которые FeatureEngineer отдаёт модели. Порядок фиксирован:
# модель работает с позициями колонок, а не с их именами.
ENGINEERED_FEATURES = [
    # Физические
    'area', 'log_area', 'area_sq', 'kitchen_area', 'kitchen_ratio',
    'rooms_clipped', 'is_studio', 'area_per_room',
    # Этажность
    'level', 'levels', 'floor_ratio', 'is_top_floor', 'is_first_floor',
    # Тип объекта
    'building_type', 'object_type',
    # Пространственные
    'geo_lat', 'geo_lon', 'dist_center_km', 'geo_cluster',
    # Временные
    'year', 'month', 'quarter', 'time_trend', 'season_sin', 'season_cos',
]


def load_data(path=None, region=None):
    """Читает CSV, оставляет один регион и отсеивает некорректные записи.

    Здесь только построчные операции, не зависящие от разбиения на
    train/test. Всё, что требует обученных статистик, — в FeatureEngineer.
    """
    path = path or settings.data_path
    region = region if region is not None else settings.region

    df = pd.read_csv(path, usecols=RAW_FEATURES + [TARGET, 'region'])
    df = df[df['region'] == region].drop(columns=['region'])

    df = df.dropna(subset=RAW_FEATURES)
    df['rooms'] = df['rooms'].replace(-1, 0)   # студии закодированы как -1
    df = df[df['kitchen_area'] > 0]
    df = df[df[TARGET] > 0]
    # Заведомо битые координаты (датасет общероссийский)
    df = df[df['geo_lat'].between(40, 70) & df['geo_lon'].between(20, 60)]

    # Капирование экстремальных цен, как в baseline-ноутбуке
    lo, hi = df[TARGET].quantile(0.01), df[TARGET].quantile(0.99)
    df[TARGET] = df[TARGET].clip(lower=lo, upper=hi)

    return df.reset_index(drop=True)


def haversine_dist(lat1, lon1, lat2, lon2):
    """Расстояние в км по формуле Haversine."""
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Сырой DataFrame -> матрица признаков в порядке ENGINEERED_FEATURES.

    transform никогда не удаляет строки: на N входных строк всегда
    возвращается N строк, иначе прогнозы не сопоставить со входом.
    """

    def __init__(self, n_clusters=12, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def _base_features(self, X):
        """Признаки, не зависящие от обученных статистик."""
        df = X.copy()

        date = pd.to_datetime(df['date'], errors='coerce')
        df['year'] = date.dt.year
        df['month'] = date.dt.month
        df['quarter'] = date.dt.quarter

        df['rooms'] = df['rooms'].replace(-1, 0)

        df['kitchen_ratio'] = (df['kitchen_area'] / df['area']).clip(0, 1)
        df['floor_ratio'] = (df['level'] / df['levels']) \
            .replace([np.inf, -np.inf], np.nan).fillna(0)
        df['is_top_floor'] = (df['level'] == df['levels']).astype(int)
        df['is_first_floor'] = (df['level'] == 1).astype(int)
        df['area_per_room'] = df['area'] / df['rooms'].replace(0, 0.5)
        df['is_studio'] = (df['rooms'] == 0).astype(int)
        df['rooms_clipped'] = df['rooms'].clip(upper=5)

        df['season_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['season_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        df['dist_center_km'] = haversine_dist(
            df['geo_lat'], df['geo_lon'], MOSCOW_CENTER[0], MOSCOW_CENTER[1]
        )
        return df

    def _fitted_features(self, df):
        """Признаки, использующие статистики из fit."""
        df['area'] = df['area'].clip(lower=self.area_min_, upper=self.area_max_)
        df['log_area'] = np.log1p(df['area'])
        df['area_sq'] = df['area'] ** 2

        df['kitchen_area'] = df['kitchen_area'].mask(
            df['kitchen_area'] <= 0, self.kitchen_area_median_
        )

        df['year'] = df['year'].fillna(self.min_year_)
        df['month'] = df['month'].fillna(1)
        df['quarter'] = df['quarter'].fillna(1)
        df['time_trend'] = (df['year'] - self.min_year_) * 12 + df['month']

        df['geo_lat'] = df['geo_lat'].fillna(self.geo_lat_median_)
        df['geo_lon'] = df['geo_lon'].fillna(self.geo_lon_median_)
        df['geo_cluster'] = self.kmeans_.predict(df[['geo_lat', 'geo_lon']].values)
        return df

    def fit(self, X, y=None):
        df = self._base_features(X)

        self.area_min_ = float(df['area'].quantile(0.01))
        self.area_max_ = float(df['area'].quantile(0.99))
        self.kitchen_area_median_ = float(
            df.loc[df['kitchen_area'] > 0, 'kitchen_area'].median()
        )
        self.geo_lat_median_ = float(df['geo_lat'].median())
        self.geo_lon_median_ = float(df['geo_lon'].median())
        self.min_year_ = int(df['year'].min())

        coords = df[['geo_lat', 'geo_lon']].fillna(
            {'geo_lat': self.geo_lat_median_, 'geo_lon': self.geo_lon_median_}
        ).values
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters, random_state=self.random_state, n_init=10
        ).fit(coords)

        # Медианы для заполнения пропусков в transform
        out = self._fitted_features(df).reindex(columns=ENGINEERED_FEATURES)
        self.feature_medians_ = out.median(numeric_only=True).fillna(0).to_dict()
        return self

    def transform(self, X):
        df = self._fitted_features(self._base_features(X))
        out = df.reindex(columns=ENGINEERED_FEATURES)
        out = out.apply(pd.to_numeric, errors='coerce').fillna(self.feature_medians_)

        assert len(out) == len(X), 'transform изменил число строк'
        return out.values

    def get_feature_names_out(self, input_features=None):
        return np.asarray(ENGINEERED_FEATURES, dtype=object)
