def test_smoke(client, good_row):
    r = client.post("/v1/predict", json=good_row)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 < body["price"]
    assert body["latency_ms"] >= 0
    assert body["model_version"]
    assert isinstance(body["price"], float)
    assert isinstance(body["model_version"], str)
    assert isinstance(body["request_id"], str)

def test_predict_handles_missing_kitchen_area(client, good_row):
    row = dict(good_row)
    del row["kitchen_area"]
    r = client.post("/v1/predict", json=row)
    assert r.status_code == 200
    
def test_predict_is_deterministic(client, good_row):
    s1 = client.post("/v1/predict", json=good_row).json()["price"]
    s2 = client.post("/v1/predict", json=good_row).json()["price"]
    assert s1 == s2

def test_batch_matches_single(client, good_row):
    single = client.post("/v1/predict", json=good_row).json()["price"]
    r = client.post("/v1/predict/batch", json={"rows": [good_row, good_row]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["prices"]) == 2
    assert body["prices"][0] == single


def test_batch_empty_is_422(client):
    assert client.post("/v1/predict/batch", json={"rows": []}).status_code == 422
