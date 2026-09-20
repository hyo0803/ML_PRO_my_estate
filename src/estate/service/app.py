import datetime
import logging
import time
import uuid
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from estate import db
from estate.config import settings

logger = logging.getLogger(__name__)


class Features(BaseModel):
    """Поля запроса = metadata['features'] бандла.

    Границы взяты с запасом от 1-го/99-го перцентилей региона 81:
    отсекают заведомо битые значения, но не режут реальные объекты.
    """
    model_config = {"extra": "forbid"}

    date: datetime.date
    geo_lat: float = Field(ge=54.0, le=57.0, description="широта, Московская область")
    geo_lon: float = Field(ge=35.0, le=41.0, description="долгота, Московская область")
    building_type: int = Field(ge=0, le=5, description="тип дома: 0 — другой, 1 — панель, 2 — монолит, 3 — кирпич, 4 — блочный, 5 — деревянный")
    object_type: int = Field(ge=1, le=11, description="1 — вторичка, 11 — новостройка")
    level: int = Field(ge=1, le=50, description="этаж квартиры")
    levels: int = Field(ge=1, le=50, description="этажей в доме")
    rooms: int = Field(ge=-1, le=10, description="комнат; -1 или 0 — студия")
    area: float = Field(gt=5, lt=1000, description="общая площадь, м²")
    # В данных бывает 0/пропуск — пайплайн заполняет медианой
    kitchen_area: float | None = Field(default=None, gt=0, lt=300, description="площадь кухни, м²")

    @model_validator(mode="after")
    def level_within_building(self):
        if self.level > self.levels:
            raise ValueError("level не может быть больше levels")
        if self.kitchen_area is not None and self.kitchen_area >= self.area:
            raise ValueError("kitchen_area должна быть меньше area")
        return self


class Prediction(BaseModel):
    price: float = Field(description="прогноз стоимости, руб.")
    model_version: str
    request_id: str
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle = joblib.load(settings.model_path)
    app.state.pipeline = bundle["pipeline"]
    app.state.meta = bundle["metadata"]
    app.state.version = bundle["metadata"]["model_version"]

    db.init()
    yield
    app.state.pipeline = None


app = FastAPI(title="estate-service", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": getattr(app.state, "version", "unknown")}


@app.get("/ready")
def ready():
    if getattr(app.state, "pipeline", None) is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready", "model_version": app.state.version}


@app.post("/v1/predict")
def predict(x: Features, bg: BackgroundTasks) -> Prediction:
    t0 = time.perf_counter()
    request_id = str(uuid.uuid4())
    version = app.state.version
    payload = x.model_dump(mode="json")  # date -> строка: и для DataFrame, и для jsonb
    frame = pd.DataFrame([payload]).reindex(columns=app.state.meta["features"])

    try:
        # Модель предсказывает log1p(цена) — возвращаем в рубли
        price = float(np.expm1(app.state.pipeline.predict(frame)[0]))
    except Exception:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.exception("prediction failed, request_id=%s", request_id)
        bg.add_task(db.save_prediction, request_id, payload, None, version, latency_ms, 500)
        raise HTTPException(status_code=500, detail="prediction failed")

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    bg.add_task(db.save_prediction, request_id, payload, price, version, latency_ms, 200)

    return Prediction(
        price=price,
        model_version=version,
        request_id=request_id,
        latency_ms=latency_ms,
    )
