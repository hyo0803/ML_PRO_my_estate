"""Настройки окружения.

Сюда попадает только то, что меняется от запуска к запуску: пути, уровень
логирования, подключение к БД. Гиперпараметры, список признаков и порог
здесь НЕ живут — они часть модели и едут внутри joblib-бандла.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Сервис
    model_path: str = 'artifact/model.joblib'
    database_url: str | None = None
    log_level: str = 'INFO'

    # Обучение
    data_path: str = 'data/all_v2.csv'
    region: int = 81

    model_config = {'env_file': '.env', 'extra': 'ignore'}


settings = Settings()
