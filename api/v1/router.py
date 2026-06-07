"""API v1 router assembly."""

from fastapi import APIRouter

from api.v1 import analysis, evaluation, portfolio

router = APIRouter(prefix="/api/v1")
router.include_router(portfolio.router, prefix="/portfolio", tags=["portfolio"])
router.include_router(analysis.router, prefix="/analysis", tags=["analysis"])
router.include_router(evaluation.router, prefix="/evaluation", tags=["evaluation"])

