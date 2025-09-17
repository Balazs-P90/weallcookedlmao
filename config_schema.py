"""
Configuration schema for End-of-World Predictor using Pydantic.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class VolatilityConfig(BaseModel):
    nuclear_pct_sd_yr: float = Field(...)
    fpi_pct_sd_yr: float = Field(...)
    conflict_score_sd_yr: float = Field(...)
    temp_trend_sd_decade: float = Field(...)

class CorrelationsConfig(BaseModel):
    vars: List[str]
    matrix: List[List[float]]

class TriggersConfig(BaseModel):
    famine_fpi_jump: Dict[str, float]
    nuclear_deploy_spike: Dict[str, float]
    temp_2c: Dict[str, float]

class ModelConfig(BaseModel):
    nuclear_alpha: float
    climate_to_food_elasticity: float
    famine_trigger_fpi: float
    conflict_base_hazard: float
    conflict_hazard_scale: float
    volatility: VolatilityConfig
    correlations: CorrelationsConfig
    triggers: TriggersConfig

class DataConfig(BaseModel):
    fao_fpi_history_csv: str
    gistemp_history_csv: str
    wars_history_csv: str
    sipri_local_csv: str
    gdelt_sample_query: str
    use_html_scrape_for_fao: bool
    fao_fpi_page: str
    fao_default_fpi: float
    temp_default_anomaly_1951_1980: float
    temp_default_trend_decade: float
    sipri_snapshot: list
    sipri_defaults: dict

class RunConfig(BaseModel):
    years_horizon: int
    monte_carlo_runs: int
    mc_chunk: int
    save_dir: str
    seed: int
    sleep_between_runs_seconds: int

class OutputConfig(BaseModel):
    save_parquet: bool
    save_csv: bool

class ScenarioThresholdsConfig(BaseModel):
    nuclear: dict
    famine_deaths: dict
    world_war_states: dict

class MetaConfig(BaseModel):
    version: str
    author: str
    notes: str

class EOWConfig(BaseModel):
    meta: MetaConfig
    model: ModelConfig
    data: DataConfig
    run: RunConfig
    output: OutputConfig
    scenario_thresholds: ScenarioThresholdsConfig
