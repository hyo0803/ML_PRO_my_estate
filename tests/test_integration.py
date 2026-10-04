import os

import psycopg
import pytest

DATABASE_URL = os.getenv("DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DATABASE_URL, reason="нужен Postgres: задайте DATABASE_URL"),
]


def fetch_log(request_id):
    with psycopg.connect(DATABASE_URL) as conn:
        return conn.execute(
            "SELECT model_version, score, status_code, features->>'building_type' "
            "FROM estate_price_predictions WHERE request_id = %s",
            (request_id,),
        ).fetchone()


def test_prediction_is_logged(client, good_row):
    body = client.post("/v1/predict", json=good_row).json()

    row = fetch_log(body["request_id"])
    assert row is not None
    assert row[0] == body["model_version"]
    assert row[1] == pytest.approx(body["price"])
    assert row[2] == 200
    assert row[3] == str(good_row["building_type"])


def test_garbage_is_logged_as_422(client, good_row):
    r = client.post("/v1/predict", json={**good_row, "area": "мусор"})
    assert r.status_code == 422

    row = fetch_log(r.json()["request_id"])
    assert row is not None
    assert row[1] is None
    assert row[2] == 422
