from fastapi import FastAPI as App
from app.api.opinion_api import router as opinion_router
from app.api.fusion_api import router as fusion_router
from app.api.outcome_api import router as outcome_router
from app.api.decision_api import router as decision_router

app = App()

@app.get("/health")
async def health():
    return {"status": "ok"}

app.include_router(opinion_router)
app.include_router(fusion_router)
app.include_router(outcome_router)
app.include_router(decision_router)
