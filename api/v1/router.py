"""API v1 router assembly."""

from fastapi import APIRouter

from api.v1 import analysis, config, evaluation, portfolio, portfolios

router = APIRouter(prefix="/api/v1")
router.include_router(portfolio.router, prefix="/portfolio", tags=["portfolio"])
router.include_router(portfolios.router, prefix="/portfolios", tags=["portfolios"])
router.include_router(analysis.router, prefix="/analysis", tags=["analysis"])
router.include_router(evaluation.router, prefix="/evaluation", tags=["evaluation"])
router.include_router(config.router, prefix="/config", tags=["config"])
