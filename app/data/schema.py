from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Sport(str, Enum):
    football = "football"
    basketball = "basketball"


class PredictionContext(str, Enum):
    live = "live"
    prematch = "prematch"
    combo_leg = "combo_leg"


class Score(BaseModel):
    home: int = Field(..., ge=0)
    away: int = Field(..., ge=0)


class MatchStats(BaseModel):
    xg_home: float = Field(..., ge=0)
    xg_away: float = Field(..., ge=0)
    shots_on_target_home: int = Field(..., ge=0)
    shots_on_target_away: int = Field(..., ge=0)
    shots_home: int = Field(..., ge=0)
    shots_away: int = Field(..., ge=0)
    corners_home: int = Field(..., ge=0)
    corners_away: int = Field(..., ge=0)
    dangerous_attacks_home: int = Field(..., ge=0)
    dangerous_attacks_away: int = Field(..., ge=0)
    possession_home: float = Field(..., ge=0, le=100)
    possession_away: float = Field(..., ge=0, le=100)
    red_cards_home: int = Field(..., ge=0)
    red_cards_away: int = Field(..., ge=0)


class Odds(BaseModel):
    over_1_5_goals: Optional[float] = Field(None, gt=1)
    over_2_5_goals: Optional[float] = Field(None, gt=1)
    under_2_5_goals: Optional[float] = Field(None, gt=1)
    btts_yes: Optional[float] = Field(None, gt=1)
    btts_no: Optional[float] = Field(None, gt=1)
    over_9_5_corners: Optional[float] = Field(None, gt=1)
    under_9_5_corners: Optional[float] = Field(None, gt=1)


class PredictionRequest(BaseModel):
    context: PredictionContext
    sport: Sport
    fixture_id: int
    league_id: int
    minute: Optional[int] = Field(None, ge=0, le=130)
    score: Optional[Score]
    stats: Optional[MatchStats]
    odds: Optional[Odds]


class PredictionResult(BaseModel):
    probability_home_win: float = Field(..., ge=0, le=1)
    probability_away_win: float = Field(..., ge=0, le=1)
    probability_draw: float = Field(..., ge=0, le=1)
    suggested_market: str
    rationale: str


class PredictionResponse(BaseModel):
    fixture_id: int
    league_id: int
    context: PredictionContext
    prediction: PredictionResult
