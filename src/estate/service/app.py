import datetime
import logging
import time
import uuid
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator
from starlette.background import BackgroundTask

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


class BatchRequest(BaseModel):
    model_config = {"extra": "forbid"}

    rows: list[Features] = Field(min_length=1, max_length=1000)


class BatchPrediction(BaseModel):
    prices: list[float] = Field(description="прогнозы стоимости, руб., в порядке rows")
    model_version: str
    request_id: str
    latency_ms: float


def load_model() -> tuple[object, dict, str]:
    """Модель из реестра MLflow по алиасу, а без MODEL_NAME из файла, как раньше."""
    if not settings.model_name:
        bundle = joblib.load(settings.model_path)
        return bundle["pipeline"], bundle["metadata"], bundle["metadata"]["model_version"]

    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mv = MlflowClient().get_model_version_by_alias(settings.model_name, settings.model_alias)
    pipeline = mlflow.sklearn.load_model(f"models:/{settings.model_name}/{mv.version}")
    meta = mlflow.artifacts.load_dict(f"runs:/{mv.run_id}/metadata.json")
    return pipeline, meta, f"{settings.model_name}-v{mv.version}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline, app.state.meta, app.state.version = load_model()

    db.init()
    yield
    app.state.pipeline = None


app = FastAPI(title="estate-service", version="1.1", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def log_validation_error(request: Request, exc: RequestValidationError):
    """422 тоже пишем в таблицу логов: features — то, что прислали, score пустой."""
    request_id = str(uuid.uuid4())
    body = exc.body if isinstance(exc.body, dict) else {"raw": str(exc.body)}
    version = getattr(app.state, "version", "unknown")
    bg = BackgroundTask(db.save_prediction, request_id, body, None, version, 0.0, 422)
    return JSONResponse(
        status_code=422,
        content={"detail": jsonable_encoder(exc.errors()), "request_id": request_id},
        background=bg,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": getattr(app.state, "version", "unknown"),
        # приходят из ConfigMap — так снаружи видно, что конфиг доехал
        "model_path": settings.model_path,
        "log_level": settings.log_level,
    }


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
    except Exception as err:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.exception("prediction failed, request_id=%s", request_id)
        bg.add_task(db.save_prediction, request_id, payload, None, version, latency_ms, 500)
        raise HTTPException(status_code=500, detail="prediction failed") from err

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    bg.add_task(db.save_prediction, request_id, payload, price, version, latency_ms, 200)

    return Prediction(
        price=price,
        model_version=version,
        request_id=request_id,
        latency_ms=latency_ms,
    )


@app.post("/v1/predict/batch")
def predict_batch(req: BatchRequest) -> BatchPrediction:
    """Все строки собираются в один DataFrame, модель вызывается один раз."""
    t0 = time.perf_counter()
    request_id = str(uuid.uuid4())
    payload = [row.model_dump(mode="json") for row in req.rows]
    frame = pd.DataFrame(payload).reindex(columns=app.state.meta["features"])

    try:
        prices = np.expm1(app.state.pipeline.predict(frame)).tolist()
    except Exception as err:
        logger.exception("batch prediction failed, request_id=%s", request_id)
        raise HTTPException(status_code=500, detail="prediction failed") from err

    return BatchPrediction(
        prices=prices,
        model_version=app.state.version,
        request_id=request_id,
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )
