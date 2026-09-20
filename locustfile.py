"""Нагрузочное тестирование сервиса.

    uv run locust -f locustfile.py --headless -u 10 -r 5 -t 60s --csv load/run10 -H http://127.0.0.1:8000
"""
from locust import HttpUser, between, task

GOOD_ROW = {
    "date": "2026-09-01",
    "geo_lat": 55.0,
    "geo_lon": 39.0,
    "building_type": 3,
    "object_type": 1,
    "level": 7,
    "levels": 9,
    "rooms": 2,
    "area": 54.0,
    "kitchen_area": 10.0,
}


class EstateUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(9)
    def predict(self):
        self.client.post("/v1/predict", json=GOOD_ROW)

    @task(1)
    def health(self):
        self.client.get("/health")
