from fastapi import APIRouter as Router
from uuid import UUID
from app.services.fusion_service import fuse_claim

router = Router()

@router.get("/fuse/{claim_id}/{domain}")
async def fuse(claim_id: UUID, domain: str):
    result = fuse_claim(claim_id, domain)
    return result
