# ------------------------- CORRELATED INPUT SAMPLER -------------------------
def sample_correlated_inputs(seed, base_inputs, config):
    """Sample correlated risk factor inputs using the config's correlation matrix and means."""
    np.random.seed(seed)
    # Extract means and covariance from config
    means = []
    stds = []
    var_names = config['model']['correlations']['vars']
    for v in var_names:
        if v == 'nuclear_growth':
            means.append(0.0)
            stds.append(config['model']['volatility']['nuclear_pct_sd_yr'])
        elif v == 'fpi_pct':
            means.append(0.0)
            stds.append(config['model']['volatility']['fpi_pct_sd_yr'])
        elif v == 'conflict_score':
            means.append(0.0)
            stds.append(config['model']['volatility']['conflict_score_sd_yr'])
        elif v == 'temp_trend':
            means.append(0.0)
            stds.append(config['model']['volatility']['temp_trend_sd_decade'])
        else:
            means.append(0.0)
            stds.append(1.0)
    cov = np.array(config['model']['correlations']['matrix'])
    # Scale covariance by stds
    cov = cov * np.outer(stds, stds)
    sample = np.random.multivariate_normal(means, cov)
    sample_dict = dict(zip(var_names, sample))
    # Build sampled input dict
    sampled = dict(base_inputs)
    # Nuclear deployed (apply growth)
    base_deployed = base_inputs.get('nuclear_deployed', config['data']['sipri_defaults']['deployed_with_forces'])
    sampled['nuclear_deployed'] = max(0.0, base_deployed * (1.0 + sample_dict.get('nuclear_growth', 0.0)))
    # FPI
    base_fpi = base_inputs.get('fao_fpi', config['data']['fao_default_fpi'])
    fpi_pct = float(sample_dict.get('fpi_pct', 0.0))
    sampled['fao_fpi'] = max(0.0, base_fpi * (1.0 + fpi_pct))
    sampled['fao_fpi_pct'] = fpi_pct
    # Conflict
    base_conflict = base_inputs.get('conflict_score', 0.0)
    sampled['conflict_score'] = base_conflict + float(sample_dict.get('conflict_score', 0.0))
    sampled['conflict_trend'] = base_inputs.get('conflict_trend', 0.0) + float(sample_dict.get('conflict_score', 0.0)) * 0.1
    # Climate
    sampled['temp_anomaly'] = base_inputs.get('temp_anomaly', config['data']['temp_default_anomaly_1951_1980'])
    sampled['temp_trend_decade'] = base_inputs.get('temp_trend_decade', config['data']['temp_default_trend_decade']) + float(sample_dict.get('temp_trend', 0.0))
    return sampled
from config_schema import EOWConfig

def load_config_from_dict(config_dict) -> EOWConfig:
    """Validate and load configuration using Pydantic schema."""
    return EOWConfig(**config_dict)
#!/usr/bin/env python3
"""
End-of-World Predictor — Version: Realistic+Advanced (fixed)
Single-file corrected implementation. Overwrites previous broken script.
"""
import os
import sys
import time
import math
import json
import signal
import logging
import argparse
import copy
from datetime import datetime
from typing import Optional

import yaml
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from multiprocessing import Pool, cpu_count

# optional

try:
    from prometheus_client import start_http_server, Gauge
    PROMETHEUS_AVAILABLE = True
except Exception:
    PROMETHEUS_AVAILABLE = False
import yaml

# Load and validate config at startup
with open('config.yaml', 'r') as f:
    raw_config = yaml.safe_load(f)
CONFIG = load_config_from_dict(raw_config)

# ------------------------- LOGGING -------------------------
logger = logging.getLogger('eow-predictor')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(fmt)
logger.addHandler(handler)

# ------------------------- DEFAULT CONFIG -------------------------
DEFAULT_CONFIG = {
    'meta': {
        'version': '2025.09.12-advanced-fixed',
        'author': 'automated-generator',
        'notes': 'Prefilled estimates and sensible defaults. Update as new data arrives.'
    },
    'model': {
        'nuclear_alpha': 1.3e-7,
        'climate_to_food_elasticity': 0.12,
        'famine_trigger_fpi': 180.0,
        'conflict_base_hazard': 1e-3,
        'conflict_hazard_scale': 0.002,
        'volatility': {
            'nuclear_pct_sd_yr': 0.03,
            'fpi_pct_sd_yr': 0.08,
            'conflict_score_sd_yr': 0.5,
            'temp_trend_sd_decade': 0.05
        },
        'correlations': {
            'vars': ['nuclear_growth', 'fpi_pct', 'conflict_score', 'temp_trend'],
            'matrix': [
                [1.0, 0.15, 0.35, 0.05],
                [0.15, 1.0, 0.25, 0.15],
                [0.35, 0.25, 1.0, 0.10],
                [0.05, 0.15, 0.10, 1.0]
            ]
        },
        'triggers': {
            'famine_fpi_jump': {'threshold_pct': 0.15, 'multiplier': 5.0},
            'nuclear_deploy_spike': {'threshold_pct': 0.1, 'multiplier': 3.0},
            'temp_2c': {'threshold_abs': 2.0, 'longterm_multiplier': 2.5}
        }
    },
    'data': {
    'sipri_local_csv': 'sipri_history.csv',
    'fao_fpi_history_csv': 'fao_fpi_history.csv',
        'gistemp_history_csv': 'gistemp_history.csv',
        'wars_history_csv': 'wars_history.csv',
        'gdelt_sample_query': 'conflict',
        'use_html_scrape_for_fao': True,
        'fao_fpi_page': 'https://www.fao.org/worldfoodsituation/foodpricesindex/en/',
        'fao_default_fpi': 130.1,
        'temp_default_anomaly_1951_1980': 1.28,
        'temp_default_trend_decade': 0.32,
        'sipri_snapshot': [
            {'Country': 'Russia', 'Deployed': 5459, 'Reserve': 1000, 'Retired': 1000, 'Total': 7459},
            {'Country': 'United States', 'Deployed': 5177, 'Reserve': 1000, 'Retired': 1000, 'Total': 7177},
            {'Country': 'China', 'Deployed': 600, 'Reserve': 0, 'Retired': 0, 'Total': 600},
            {'Country': 'France', 'Deployed': 290, 'Reserve': 0, 'Retired': 0, 'Total': 290},
            {'Country': 'United Kingdom', 'Deployed': 225, 'Reserve': 0, 'Retired': 0, 'Total': 225},
            {'Country': 'Pakistan', 'Deployed': 170, 'Reserve': 0, 'Retired': 0, 'Total': 170},
            {'Country': 'India', 'Deployed': 180, 'Reserve': 0, 'Retired': 0, 'Total': 180},
            {'Country': 'Israel', 'Deployed': 90, 'Reserve': 0, 'Retired': 0, 'Total': 90},
            {'Country': 'North Korea', 'Deployed': 50, 'Reserve': 0, 'Retired': 0, 'Total': 50}
        ],
        'sipri_defaults': {
            'total_inventory': 12241,
            'stockpiles_potential_use': 9614,
            'deployed_with_forces': 3912
        }
    },
    'run': {
        'years_horizon': 100,
        'monte_carlo_runs': 20000,
        'mc_chunk': 2000,
        'save_dir': 'results',
        'seed': 42,
        'sleep_between_runs_seconds': 3
    },
    'output': {
        'save_parquet': True,
        'save_csv': True
    },
    'scenario_thresholds': {
        'nuclear': {'1K': 10, '1M': 1000, '100M': 100000},
        'famine_deaths': {'1K': 1000, '1M': 1000000, '10M': 10000000},
        'world_war_states': {'regional': 3, 'global': 10}
    }
}

# ------------------------- HELPERS -------------------------
def make_session(retries=3, backoff_factor=0.3, status_forcelist=(500,502,504)):
    s = requests.Session()
    retry = Retry(total=retries, read=retries, connect=retries, backoff_factor=backoff_factor,
                  status_forcelist=status_forcelist, allowed_methods=frozenset(['GET','POST']))
    adapter = HTTPAdapter(max_retries=retry)
    s.mount('https://', adapter)
    s.mount('http://', adapter)
    return s

HTTP = make_session()

def safe_get(url, params=None, timeout=15):
    try:
        r = HTTP.get(url, params=params, timeout=timeout, headers={"User-Agent":"eow-predictor/1.0"})
        r.raise_for_status()
        return r
    except Exception as e:
        logger.debug('HTTP error %s : %s', url, e)
        return None

# ------------------------- DATA FETCHERS / PARSERS -------------------------
def fetch_acled_events(token: Optional[str]=None, limit=10000):
    token = token or os.getenv('ACLED_TOKEN')
    if not token:
        logger.info('ACLED token not provided. Skipping ACLED fetch.')
        return None
    url = 'https://api.acleddata.com/acled/read'
    params = {'limit': limit, 'email': token}
    r = safe_get(url, params=params)
    if r is None:
        logger.warning('ACLED fetch failed.')
        return None
    try:
        j = r.json()
        events = j.get('data') or j.get('results') or []
        df = pd.DataFrame(events)
        return df
    except Exception as e:
        logger.warning('Failed to parse ACLED response: %s', e)
        return None

def fetch_gdelt_events(keyword='conflict', startdatetime=None, enddatetime=None):
    url = 'https://api.gdeltproject.org/api/v2/doc/doc'
    params = {'query': keyword, 'mode': 'artlist', 'format': 'JSON'}
    if startdatetime:
        params['startdatetime'] = startdatetime
    if enddatetime:
        params['enddatetime'] = enddatetime
    r = safe_get(url, params=params)
    if r is None:
        logger.info('GDELT fetch failed or blocked.')
        return None
    try:
        j = r.json()
        articles = j.get('articles', []) or j.get('data', [])
        df = pd.DataFrame(articles)
        return df
    except Exception as e:
        logger.warning('Failed to parse GDELT response: %s', e)
        return None

def compute_climate_index(gistemp_df=None):
    if gistemp_df is None or gistemp_df.empty:
        return (CONFIG.data.temp_default_anomaly_1951_1980,
                CONFIG.data.temp_default_trend_decade)
    try:
        # Support different formats: Year,J-D or first numeric column as anomaly
        if 'J-D' in gistemp_df.columns:
            series = pd.to_numeric(gistemp_df['J-D'].replace('***', np.nan), errors='coerce').dropna()
            years = gistemp_df['Year'].astype(int)
            s = pd.Series(series.values, index=years.values)
        else:
            # fallback: take last numeric column
            for c in gistemp_df.columns[::-1]:
                try:
                    series = pd.to_numeric(gistemp_df[c].replace('***', np.nan), errors='coerce').dropna()
                    s = pd.Series(series.values, index=range(len(series)))
                    break
                except Exception:
                    continue
        recent = s.dropna().iloc[-20:]
        if len(recent) < 2:
            return float(s.dropna().iloc[-1]), CONFIG.data.temp_default_trend_decade
        trend = (recent.iloc[-1] - recent.iloc[0]) / (len(recent)-1) * 10.0
        current_anomaly = float(recent.iloc[-1])
        return current_anomaly, float(trend)
    except Exception as e:
        logger.warning('Error computing climate index: %s', e)
    return (CONFIG.data.temp_default_anomaly_1951_1980,
        CONFIG.data.temp_default_trend_decade)

def compute_food_price_index(fao_df=None):
    if fao_df is None or fao_df.empty:
        return CONFIG.data.fao_default_fpi, 0.0
    try:
        # look for FPI column or numeric series
        possible = None
        for c in fao_df.columns[::-1]:
            if any(sub.lower() in c.lower() for sub in ('fpi','food','index')):
                possible = c
                break
        if possible is None:
            possible = fao_df.columns[-1]
        series = pd.to_numeric(fao_df[possible].dropna(), errors='coerce').dropna()
        if series.empty:
            return CONFIG.data.fao_default_fpi, 0.0
        latest = float(series.iloc[-1])
        pct12 = float((latest - series.iloc[-13]) / series.iloc[-13]) if len(series) > 13 else 0.0
        return latest, pct12
    except Exception as e:
        logger.warning('Error computing FAO FPI: %s', e)
        return CONFIG.data.fao_default_fpi, 0.0

# ------------------------- RISK MATH -------------------------
def hazard_time_from_rate(rate_per_year):
    if rate_per_year <= 0:
        return math.inf, math.inf
    median = math.log(2) / rate_per_year
    mean = 1.0 / rate_per_year
    return median, mean

def nuclear_annual_probability(deployed_warheads, alpha):
    deployed = max(0.0, float(deployed_warheads))
    a = float(alpha)
    p = 1.0 - math.exp(-a * deployed)
    return min(max(p, 0.0), 0.999999)

def climate_escalation_years_to_threshold(current_anomaly, trend_per_decade, threshold=2.0):
    if trend_per_decade <= 0:
        return math.inf
    trend_per_year = trend_per_decade / 10.0
    years = (threshold - current_anomaly) / trend_per_year
    return years if years > 0 else 0.0

# ------------------------- DATA LOAD/BUILD helpers -------------------------
def load_or_build_sipri(path=None):
    p = path or CONFIG.data.sipri_local_csv
    if os.path.exists(p):
        try:
            df = pd.read_csv(p)
            logger.info('Loaded SIPRI CSV: %s', p)
            return df
        except Exception as e:
            logger.warning('Failed load SIPRI CSV %s: %s', p, e)
    # build from snapshot
    df = pd.DataFrame(CONFIG.data.sipri_snapshot)
    try:
        df.to_csv(p, index=False)
        logger.info('Wrote cached SIPRI snapshot to %s', p)
    except Exception:
        pass
    return df

def load_or_build_fao_history(path=None):
    p = path or CONFIG.data.fao_fpi_history_csv
    if os.path.exists(p):
        try:
            df = pd.read_csv(p, parse_dates=[0])
            logger.info('Loaded FAO FPI history: %s', p)
            return df
        except Exception as e:
            logger.warning('Failed load FAO history %s: %s', p, e)
    # fallback: single-row default
    df = pd.DataFrame({'date':[pd.Timestamp.now().normalize()], 'FPI':[CONFIG.data.fao_default_fpi]})
    try:
        df.to_csv(p, index=False)
    except Exception:
        pass
    return df

def load_or_build_gistemp(path=None):
    p = path or CONFIG.data.gistemp_history_csv
    if os.path.exists(p):
        try:
            df = pd.read_csv(p)
            logger.info('Loaded GISTEMP history: %s', p)
            return df
        except Exception as e:
            logger.warning('Failed load GISTEMP %s: %s', p, e)
    # fallback to defaults as small df
    df = pd.DataFrame({'Year':[2024], 'J-D':[CONFIG.data.temp_default_anomaly_1951_1980]})
    try:
        df.to_csv(p, index=False)
    except Exception:
        pass
    return df

def load_or_build_wars_history(path=None):
    p = path or CONFIG.data.wars_history_csv
    if os.path.exists(p):
        try:
            df = pd.read_csv(p, parse_dates=['start_date','end_date'], infer_datetime_format=True)
            logger.info('Loaded wars history: %s', p)
            return df
        except Exception as e:
            logger.warning('Failed load wars csv %s: %s', p, e)
    # fallback empty
    df = pd.DataFrame(columns=['name','country','start_date','end_date','fatalities_est'])
    try:
        df.to_csv(p, index=False)
    except Exception:
        pass
    return df

# ------------------------- PROXY CONFLICT SCORE (no ACLED) -------------------------
def compute_proxy_conflict_score(gdelt_df=None, wars_df=None, fpi_series=None, _temp_anomaly=None, sipri_df=None):
    score = 0.0
    trend = 0.0
    # GDELT: count articles as proxy
    try:
        if gdelt_df is not None and not gdelt_df.empty:
            datecols = [c for c in gdelt_df.columns if 'date' in c.lower() or 'seendate' in c.lower()]
            # Example: count articles as proxy for conflict
            score += min(5.0, len(gdelt_df) / 1000.0)
    except Exception:
        pass
    # Wars: count recent wars as proxy
    try:
        if wars_df is not None and not wars_df.empty:
            recent = wars_df[wars_df['end_date'] > pd.Timestamp.now() - pd.Timedelta(days=365*10)]
            score += min(3.0, len(recent) / 2.0)
    except Exception:
        pass
    # FPI: high food prices can increase conflict
    try:
        if fpi_series is not None:
            if hasattr(fpi_series, 'iloc'):
                fpi_val = float(fpi_series.iloc[-1])
            else:
                fpi_val = float(fpi_series)
            if fpi_val > 150:
                score += 1.0
    except Exception:
        pass
    # Temperature anomaly: higher temp can increase conflict
    try:
        if _temp_anomaly is not None and _temp_anomaly > 1.5:
            score += 0.5
    except Exception:
        pass
    # SIPRI: more deployed warheads can increase conflict
    try:
        if sipri_df is not None and not sipri_df.empty:
            deployed = sipri_df['Deployed'].sum()
            if deployed > 5000:
                score += 0.5
    except Exception:
        pass
    # Trend: simple proxy (could be improved)
    trend = 0.1 * score
    return score, trend
    fpi_pct = float(sample.get('fpi_pct', 0.0))
    sampled['fao_fpi'] = max(0.0, base_fpi * (1.0 + fpi_pct))
    sampled['fao_fpi_pct'] = fpi_pct
    base_conflict = base_inputs.get('conflict_score', 0.0)
    sampled['conflict_score'] = base_conflict + float(sample.get('conflict_score', 0.0))
    sampled['conflict_trend'] = base_inputs.get('conflict_trend', 0.0) + float(sample.get('conflict_score', 0.0)) * 0.1
    sampled['temp_anomaly'] = base_inputs.get('temp_anomaly', config['data']['temp_default_anomaly_1951_1980'])
    sampled['temp_trend_decade'] = base_inputs.get('temp_trend_decade', config['data']['temp_default_trend_decade']) + float(sample.get('temp_trend', 0.0))
    return sampled

# ------------------------- EVENT TRIGGERS -------------------------
def apply_event_triggers(sampled, config):
    multiplier = 1.0
    triggers = config['model']['triggers']
    try:
        if sampled.get('fao_fpi_pct', 0.0) > triggers['famine_fpi_jump']['threshold_pct']:
            multiplier *= triggers['famine_fpi_jump']['multiplier']
    except Exception:
        pass
    try:
        base = config['data']['sipri_defaults']['stockpiles_potential_use']
        if (sampled.get('nuclear_deployed', base) - base) / base > triggers['nuclear_deploy_spike']['threshold_pct']:
            multiplier *= triggers['nuclear_deploy_spike']['multiplier']
    except Exception:
        pass
    try:
        if sampled.get('temp_anomaly', 0.0) >= triggers['temp_2c']['threshold_abs']:
            multiplier *= triggers['temp_2c']['longterm_multiplier']
    except Exception:
        pass
    return multiplier

# ------------------------- SINGLE RUN -------------------------
def single_simulation_from_base(seed, base_inputs, config):
    from risk_model import RiskModel
    from config_schema import EOWConfig
    # If config is a dict, reconstruct EOWConfig
    if isinstance(config, dict):
        config_obj = EOWConfig(**config)
    else:
        config_obj = config
    s = sample_correlated_inputs(seed, base_inputs, config)
    trigger_mul = apply_event_triggers(s, config)
    model = RiskModel(config_obj)
    # Nuclear risk
    p_nuclear = model.nuclear_risk(s['nuclear_deployed'], s['conflict_score']) * trigger_mul
    p_nuclear = min(p_nuclear, 0.9999)
    # Famine risk (10-year compounded, with explainability)
    p_famine, famine_explain = model.famine_risk(s['fao_fpi'], s['conflict_score'], s['temp_anomaly'], years=10, explain=True)
    p_famine = min(p_famine * trigger_mul, 0.9999)
    # Conflict risk (hazard, with famine feedback)
    war_hazard = model.conflict_risk(s['conflict_score'], s.get('conflict_trend', 0.0), famine_risk=p_famine)
    war_hazard = max(1e-8, war_hazard)
    # Convert probabilities to hazard rates
    nuclear_hazard = -math.log(1 - p_nuclear) if p_nuclear < 0.9999 else 10.0
    famine_hazard = -math.log(1 - p_famine) if p_famine < 0.9999 else 10.0
    nuclear_median, _ = hazard_time_from_rate(nuclear_hazard)
    famine_median, _ = hazard_time_from_rate(famine_hazard)
    war_median, _ = hazard_time_from_rate(war_hazard)
    # Climate risk (years to 2C)
    years_to_2C = model.climate_risk(s['temp_anomaly'], s['temp_trend_decade'])
    thresholds = config['scenario_thresholds']
    nuclear_1M_breach = s['nuclear_deployed'] >= thresholds['nuclear']['1M']
    famine_10M_breach = (p_famine > 0.5) and (s['fao_fpi'] > config['model']['famine_trigger_fpi'] * 1.2)
    world_war_global = (s['conflict_score'] > 4.0) and (war_median < 10)
    return {
        'p_nuclear': p_nuclear,
        'nuclear_median_years': nuclear_median,
        'p_famine': p_famine,
        'famine_median_years': famine_median,
        'war_median_years': war_median,
        'years_to_2C': years_to_2C,
        'deployed': s['nuclear_deployed'],
        'fpi': s['fao_fpi'],
        'temp': s['temp_anomaly'],
        'nuclear_1M_breach': nuclear_1M_breach,
        'famine_10M_breach': famine_10M_breach,
        'world_war_global': world_war_global
    }

# ------------------------- PARALLEL MC -------------------------
def run_mc_parallel(base_inputs, config, runs=None):
    # Always pass config as dict to subprocesses
    if not isinstance(config, dict):
        config_dict = config.dict()
    else:
        config_dict = config
    runs = int(runs or config_dict['run']['monte_carlo_runs'])
    chunk = int(config_dict['run'].get('mc_chunk', max(1000, runs // 20)))
    rng = np.random.default_rng(config_dict['run'].get('seed', None))
    seeds = rng.integers(0, 2**32 - 2, size=runs, dtype=np.uint64)
    cpu = max(1, min(cpu_count() - 1, runs))
    logger.info('Starting MC: runs=%d, chunk=%d, workers=%d', runs, chunk, cpu)
    def _chunked_indices(total, size):
        for i in range(0, total, size):
            yield i, min(total, i + size)
    results = []
    with Pool(processes=cpu) as pool:
        for a, b in _chunked_indices(runs, chunk):
            batch = seeds[a:b].tolist()
            jobs = [ (int(int(s) & 0xFFFFFFFF), base_inputs, config_dict) for s in batch ]
            batch_res = pool.starmap(single_simulation_from_base, jobs)
            results.extend(batch_res)
    df = pd.DataFrame(results)
    return df

# ------------------------- INPUT ASSEMBLY -------------------------
def assemble_base_inputs(_acled_df=None, gdelt_df=None, fao_df=None, gistemp_df=None, sipri_df=None, wars_df=None):
    if sipri_df is not None and not sipri_df.empty:
        cols = [c for c in sipri_df.columns if 'deployed' in c.lower() or 'total' in c.lower()]
        deployed_total = None
        for c in cols:
            try:
                deployed_total = float(sipri_df[c].astype(float).sum())
                break
            except Exception:
                continue
        if deployed_total is None:
            try:
                deployed_total = float(CONFIG.data.sipri_snapshot[0]['Deployed'])
            except Exception:
                deployed_total = float(CONFIG.data.sipri_defaults['deployed_with_forces'])
    else:
        deployed_total = sum([int(x.get('Deployed',0)) for x in CONFIG.data.sipri_snapshot])

    fapi = CONFIG.data.fao_default_fpi
    fpi_pct = 0.0
    if fao_df is not None and not fao_df.empty:
        # try to find FPI series
        series = None
        for c in fao_df.columns[::-1]:
            try:
                series = pd.to_numeric(fao_df[c].dropna(), errors='coerce').dropna()
                if not series.empty:
                    fapi = float(series.iloc[-1])
                    if len(series) > 12:
                        fpi_pct = float((series.iloc[-1] - series.iloc[-13]) / max(1.0, series.iloc[-13]))
                    break
            except Exception:
                continue

    temp_anom, temp_trend = compute_climate_index(gistemp_df)

    wars = wars_df if (wars_df is not None and not wars_df.empty) else None

    # pass fpi_series for proxy computation if available
    fpi_series = None
    if fao_df is not None and not fao_df.empty:
        for c in fao_df.columns[::-1]:
            try:
                s = pd.to_numeric(fao_df[c].dropna(), errors='coerce').dropna()
                if not s.empty:
                    fpi_series = s
                    break
            except Exception:
                continue

    conflict_score, conflict_trend = compute_proxy_conflict_score(gdelt_df, wars, fpi_series, temp_anom, sipri_df)

    return {
        'nuclear_deployed': float(deployed_total),
        'conflict_score': float(conflict_score),
        'conflict_trend': float(conflict_trend),
        'fao_fpi': float(fapi),
        'fao_fpi_pct': float(fpi_pct),
        'temp_anomaly': float(temp_anom),
        'temp_trend_decade': float(temp_trend)
    }

# ------------------------- OUTPUT SUMMARY -------------------------
def summarize_mc(df: pd.DataFrame, config) -> tuple:
    """
    Summarize Monte Carlo results with percentiles, confidence intervals, scenario analysis, and uncertainty quantification.
    Args:
        df: DataFrame of MC results
        config: model config
    Returns:
        out: dict of summary statistics
        freq: dict of event frequencies
    """
    def pct_range(series, p_low=1, p_high=99):
        """Return (low, high) percentiles for a series, ignoring inf/nan."""
        s = series.replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            return math.inf, math.inf
        return float(np.percentile(s, p_low)), float(np.percentile(s, p_high))

    def ci(series, alpha=0.05):
        """Return (lower, upper) confidence interval for the mean using bootstrapping."""
        s = series.replace([np.inf, -np.inf], np.nan).dropna().values
        if len(s) == 0:
            return (math.nan, math.nan)
        boot_means = [np.mean(np.random.choice(s, size=len(s), replace=True)) for _ in range(1000)]
        lower = np.percentile(boot_means, 100 * alpha / 2)
        upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
        return float(lower), float(upper)

    out = {}
    out['nuclear_median_1_99'] = pct_range(df['nuclear_median_years'])
    out['nuclear_median_CI'] = ci(df['nuclear_median_years'])
    out['famine_median_1_99'] = pct_range(df['famine_median_years'])
    out['famine_median_CI'] = ci(df['famine_median_years'])
    out['war_median_1_99'] = pct_range(df['war_median_years'])
    out['war_median_CI'] = ci(df['war_median_years'])
    out['years_to_2C_1_99'] = pct_range(df['years_to_2C']) if 'years_to_2C' in df.columns else (math.inf, math.inf)
    out['years_to_2C_CI'] = ci(df['years_to_2C']) if 'years_to_2C' in df.columns else (math.nan, math.nan)

    # Probability of event within 10 years (relevant horizon)
    def prob_within_n_years(median_years, n=10):
        # Assume exponential distribution: P = 1 - exp(-n/mean)
        # Use median to estimate mean: mean ≈ median / ln(2)
        if median_years <= 0 or median_years == math.inf:
            return 0.0
        mean = median_years / math.log(2)
        return 1.0 - math.exp(-n / mean)

    out['prob_nuclear_war_10y'] = prob_within_n_years(df['nuclear_median_years'].median(), 10)
    out['prob_famine_10y'] = prob_within_n_years(df['famine_median_years'].median(), 10)
    out['prob_war_10y'] = prob_within_n_years(df['war_median_years'].median(), 10)
    # Scenario analysis: probability of at least one major event in 10 years
    out['prob_any_major_event_10y'] = 1.0 - (
        (1.0 - out['prob_nuclear_war_10y']) *
        (1.0 - out['prob_famine_10y']) *
        (1.0 - out['prob_war_10y'])
    )

    freq = {}
    for col in ['nuclear_1M_breach', 'famine_10M_breach', 'world_war_global']:
        if col in df.columns:
            freq[col] = float(df[col].mean())
    # Uncertainty quantification: add stddevs
    out['nuclear_median_std'] = float(df['nuclear_median_years'].std())
    out['famine_median_std'] = float(df['famine_median_years'].std())
    out['war_median_std'] = float(df['war_median_years'].std())
    out['years_to_2C_std'] = float(df['years_to_2C'].std()) if 'years_to_2C' in df.columns else float('nan')
    return out, freq

# ------------------------- CONTROLLED RUN -------------------------
SHUTDOWN = False
def _signal_handler(sig, frame):
    global SHUTDOWN
    logger.info('Shutdown signal received: %s', sig)
    SHUTDOWN = True

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

def ensure_save_dir(path):
    os.makedirs(path, exist_ok=True)

def start_prometheus(port):
    if not PROMETHEUS_AVAILABLE:
        logger.info('Prometheus client not available. Skipping metrics server.')
        return None
    try:
        start_http_server(port)
        g = Gauge('eow_mc_runs_total', 'Total Monte Carlo runs completed')
        logger.info('Prometheus metrics started on port %d', port)
        return g
    except Exception as e:
        logger.warning('Failed to start prometheus server: %s', e)
        return None

# ------------------------- MAIN -------------------------
def deep_update(d, u):
    for k, v in (u or {}).items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--acled_token', type=str, default=None)
    parser.add_argument('--sipri_csv', type=str, default=None)
    parser.add_argument('--fao_csv', type=str, default=None)
    parser.add_argument('--gistemp_csv', type=str, default=None)
    parser.add_argument('--wars_csv', type=str, default=None)
    parser.add_argument('--daemon', action='store_true', help='Run continuously every sleep interval')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    parser.add_argument('--runs', type=int, default=None)
    parser.add_argument('--scenario', type=str, default=None, help='Run a custom scenario (JSON or YAML string)')
    parser.add_argument('--backtest', action='store_true', help='Run backtesting/validation against historical events')
    parser.add_argument('--explain', action='store_true', help='Output feature importances for ensemble/ML models')
    args = parser.parse_args(argv)

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            user_cfg = yaml.safe_load(f) or {}
            deep_update(cfg, user_cfg)
        cfg = CONFIG  # Use CONFIG instead of DEFAULT_CONFIG

    # ensure server exists
    cfg.setdefault('server', DEFAULT_CONFIG.get('server', {}))
    save_dir = cfg['run']['save_dir']
    ensure_save_dir(save_dir)

    fh = logging.FileHandler(os.path.join(save_dir, 'eow_predictor.log'))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    prom_gauge = None
    if PROMETHEUS_AVAILABLE:
        prom_gauge = start_prometheus(cfg['server'].get('prometheus_port', 8000))

    def run_scenario(scenario_inputs=None):
        # Accepts a dict of scenario overrides for base_inputs
        token = args.acled_token or os.getenv('ACLED_TOKEN')
        acled_df = None
        if token:
            acled_df = fetch_acled_events(token=token)
        gdelt_df = None
        if acled_df is None:
            gdelt_df = fetch_gdelt_events(keyword=cfg['data'].get('gdelt_sample_query','conflict'))

        fao_df = None
        if args.fao_csv and os.path.exists(args.fao_csv):
            try:
                fao_df = pd.read_csv(args.fao_csv)
            except Exception:
                fao_df = None
        if fao_df is None:
            fao_df = load_or_build_fao_history(args.fao_csv)

        gistemp_df = None
        if args.gistemp_csv and os.path.exists(args.gistemp_csv):
            try:
                gistemp_df = pd.read_csv(args.gistemp_csv)
            except Exception:
                gistemp_df = None
        if gistemp_df is None:
            gistemp_df = load_or_build_gistemp(args.gistemp_csv)

        wars_df = None
        if args.wars_csv and os.path.exists(args.wars_csv):
            try:
                wars_df = pd.read_csv(args.wars_csv, parse_dates=['start_date','end_date'])
            except Exception:
                wars_df = None
        if wars_df is None:
            wars_df = load_or_build_wars_history(args.wars_csv)

        sipri_df = None
        if args.sipri_csv and os.path.exists(args.sipri_csv):
            try:
                sipri_df = pd.read_csv(args.sipri_csv)
            except Exception:
                sipri_df = None
        if sipri_df is None:
            sipri_df = load_or_build_sipri(args.sipri_csv)

        base_inputs = assemble_base_inputs(acled_df, gdelt_df, fao_df, gistemp_df, sipri_df, wars_df)
        if scenario_inputs:
            base_inputs.update(scenario_inputs)
        logger.info('Inputs: %s', json.dumps(base_inputs, default=str))

        runs = args.runs or cfg['run']['monte_carlo_runs']
        mc_df = run_mc_parallel(base_inputs, cfg, runs=runs)

        ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        fn_base = os.path.join(save_dir, f'mc_results_{ts}')
        try:
            if cfg['output'].get('save_parquet', True):
                mc_df.to_parquet(fn_base + '.parquet', index=False)
            if cfg['output'].get('save_csv', True):
                mc_df.to_csv(fn_base + '.csv', index=False)
            logger.info('Saved MC results to %s(.parquet/.csv)', fn_base)
        except Exception as e:
            logger.warning('Failed to save results: %s', e)

        summary, freqs = summarize_mc(mc_df, cfg)
        logger.info('Summary: %s', json.dumps(summary))
        logger.info('Frequencies: %s', json.dumps(freqs))

        if args.explain:
            try:
                from ensemble_models import HyperEnsembleRiskModel
                model = HyperEnsembleRiskModel()
                if hasattr(model.ensemble, 'feature_importances_'):
                    logger.info('Ensemble feature importances: %s', model.ensemble.feature_importances_)
                elif hasattr(model.ensemble, 'estimators_'):
                    # Output feature importances for each estimator
                    for name, est in zip(model.ensemble.named_estimators.keys(), model.ensemble.estimators_):
                        if hasattr(est, 'feature_importances_'):
                            logger.info('Estimator %s importances: %s', name, est.feature_importances_)
            except Exception as e:
                logger.warning('Explainability failed: %s', e)

        if prom_gauge is not None:
            try:
                prom_gauge.inc()
            except Exception:
                pass

        return mc_df, summary, freqs

    # Scenario explorer: run custom scenario if provided
    if args.scenario:
        try:
            scenario_arg = args.scenario.strip()
            if scenario_arg.startswith('@'):
                # Load scenario from file
                scenario_path = scenario_arg[1:]
                with open(scenario_path, 'r') as f:
                    scenario_inputs = json.load(f)
            elif scenario_arg.startswith('{'):
                scenario_inputs = json.loads(scenario_arg)
            else:
                scenario_inputs = yaml.safe_load(scenario_arg)
        except Exception as e:
            logger.error('Failed to parse scenario: %s', e)
            scenario_inputs = None
        run_scenario(scenario_inputs)
        return

    # Backtesting/validation mode
    if args.backtest:
        logger.info('Running backtesting/validation...')
        # Placeholder: implement backtesting logic here
        # For each historical event, set up scenario_inputs and compare model output to reality
        # ...
        logger.info('Backtesting complete (stub).')
        return

    # Main run loop (daemon or once)
    while True:
        if SHUTDOWN:
            logger.info('Shutdown flag set. Exiting run loop.')
            break
        run_scenario()
        if args.once:
            logger.info('Completed single run, exiting (--once).')
            break
        sleep_s = int(cfg['run'].get('sleep_between_runs_seconds', cfg['run'].get('sleep_between_runs_seconds', 3600)))
        logger.info('Sleeping %d seconds before next run. Send SIGINT or SIGTERM to stop.', sleep_s)
        for i in range(sleep_s):
            if SHUTDOWN:
                break
            time.sleep(1)
        if SHUTDOWN:
            logger.info('Shutdown during sleep. Exiting.')
            break

if __name__ == '__main__':
    main()
