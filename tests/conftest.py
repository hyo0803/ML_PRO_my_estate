import pytest
from fastapi.testclient import TestClient

from estate.service.app import app

@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture()
def good_row():
    return {
        "date": '2026-09-01',
        "geo_lat": 55.0,
        "geo_lon": 39.0,
        "building_type": 3, #тип дома: 0 — другой, 1 — панель, 2 — монолит, 3 — кирпич, 4 — блочный, 5 — деревянный
        "object_type": 1, #1 — вторичка, 11 — новостройка
        "level": 7, #этаж квартиры
        "levels": 9, #этажей в доме
        "rooms": 2, #комнат; -1 или 0 — студия
        "area": 54.0, #общая площадь, м²
        "kitchen_area": 10.0, #площадь кухни, м²
    }