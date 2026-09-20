import psycopg
from psycopg.types.json import Json

from estate.config import settings

DDL = """
CREATE TABLE IF NOT EXISTS estate_price_predictions (

    request_id      uuid PRIMARY KEY,
    ts      timestamptz NOT NULL DEFAULT now(),
    model_version       text NOT NULL,
    features        jsonb NOT NULL,
    score       double precision,
    latency_ms real,
    status_code smallint NOT NULL
)
"""

def init() -> None:
    if not settings.database_url:
        return
    with psycopg.connect(settings.database_url) as conn:
        conn.execute(DDL)


def save_prediction(
    request_id : str, 
    features : dict, 
    score : float | None, 
    model_version : str, 
    latency_ms : float,
    status_code : int
    ) -> None:
    if not settings.database_url:
        return
    with psycopg.connect(settings.database_url) as conn:
            conn.execute(
            "INSERT INTO estate_price_predictions (request_id, model_version, features, score, latency_ms, status_code) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (request_id, model_version, Json(features), score, latency_ms, status_code),
        )