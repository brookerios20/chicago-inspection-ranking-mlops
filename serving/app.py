import json                                   # read version.json
from pathlib import Path                      # build paths relative to this file
from contextlib import asynccontextmanager    # only if you use FastAPI's lifespan hook

import pandas as pd                           # request -> DataFrame for the model
import skops.io as sio                        # load model.skops

from fastapi import FastAPI, HTTPException    # app + clean error responses
from pydantic import BaseModel, ConfigDict, Field
      


MODEL_DIR = Path(__file__).parent / "model"
TRUSTED = ["numpy.dtype"]   # your reviewed list

state = {}   # holds loaded objects for the endpoints

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- runs once at startup ---
    state["model"] = sio.load(MODEL_DIR / "model.skops", trusted=TRUSTED)
    state["version"] = json.loads((MODEL_DIR / "version.json").read_text())
    state["features"] = state["version"]["features"]
    yield
    # --- runs once at shutdown (nothing to clean up here) ---
    state.clear()

app = FastAPI(lifespan=lifespan)

# ---------- Request ----------

class Facility(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facility_id: str                          # client's ID, not a model feature

    # categorical
    inspection_group: str                     # e.g. "canvass"
    facility_group: str                       # e.g. "restaurant"
    risk: str                                 # e.g. "high"
    prev_result: str | None = None            # e.g. "Pass", "Pass w/ Conditions"

    # numeric
    prior_fails: float
    n_prior: float
    days_since_prev: float | None = None
    prev_n_violations: float | None = None
    zip_fail_rate_365d: float
    zip_complaints_90d: float


class RankRequest(BaseModel):
    facilities: list[Facility] = Field(min_length=1)   # batch: at least one facility


# ---------- Response ----------

class ScoredFacility(BaseModel):
    facility_id: str
    score: float      # predicted probability of failing the inspection
    rank: int         # 1 = highest risk


class RankResponse(BaseModel):
    model_version: str          # from version.json (e.g. semver), so every response is traceable
    results: list[ScoredFacility]

@app.get("/health")
def health():
    return {"status": "ok", **state["version"]}

NUMERIC = ["prior_fails", "n_prior", "days_since_prev", "prev_n_violations",
           "zip_fail_rate_365d", "zip_complaints_90d"]
           
@app.post("/predict", response_model=RankResponse)
def predict(req: RankRequest):
    try:
        # request -> DataFrame, columns in training order, facility_id dropped
        df = pd.DataFrame([f.model_dump() for f in req.facilities])[state["features"]]
        # force numeric dtype so an all-None column becomes NaN floats, not object
        df[NUMERIC] = df[NUMERIC].astype(float)

        scores = state["model"].predict_proba(df)[:, 1]   # P(fail) for each facility
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model could not score input: {e}")

    # pair each score with its facility_id, sort highest risk first
    ranked = sorted(
        zip([f.facility_id for f in req.facilities], scores),
        key=lambda pair: pair[1],
        reverse=True,
    )

    return RankResponse(
        model_version=state["version"]["semver"],
        results=[
            ScoredFacility(facility_id=fid, score=float(s), rank=i + 1)
            for i, (fid, s) in enumerate(ranked)
        ],
    )