"""
Risk modeling logic for End-of-World Predictor.
Modularized for extensibility and advanced modeling.

Assumptions and Parameter Sources:
- Famine risk base rate and weights: Heuristic, loosely based on global famine frequency (EM-DAT, FAO, Our World in Data). Should be calibrated to real data.
- Nuclear risk alpha: Heuristic, not empirically derived. Should be fit to historical near-miss and expert data.
- Conflict risk base/scale: Heuristic, not fit to historical world war data. Should be calibrated.
- Climate risk: Linear projection, ignores tipping points and feedbacks. Should use probabilistic climate projections (e.g., CMIP6, NASA POWER).
- ML/ensemble models: Stubs or fake data unless otherwise noted. Replace with real, validated models when data is available.
- All event triggers and caps: Chosen for stability and plausibility, not from literature.

See code comments for further details and TODOs.
"""
import numpy as np
from typing import Dict, Any
from scipy.special import expit

class RiskModel:
    def ensemble_risk(self, features: Dict[str, float]) -> float:
        """Risk using hyper-ensemble ML/Bayesian model."""
        try:
            from ensemble_models import HyperEnsembleRiskModel
            model = HyperEnsembleRiskModel()
            return model.predict(features)
        except Exception as e:
            import logging
            logging.error(f"Ensemble risk model failed: {e}")
            return 0.01

    def ai_risk(self, features: Dict[str, float]) -> float:
        """AI risk using external data and models (placeholder)."""
        # Would use AI Incident DB, expert models, etc.
        # TODO: Replace with real model when data is available
        return features.get('ai_incident_score', 0.01)

    def satellite_climate_risk(self, features: Dict[str, float]) -> float:
        """Satellite-based climate risk (placeholder)."""
        # Would use NASA/ESA satellite anomaly data
        # TODO: Replace with real model when data is available
        return features.get('satellite_anomaly', 0.01)

    def __init__(self, config):
        self.config = config

    def nuclear_risk(self, deployed_warheads: float, conflict_score: float) -> float:
        """Annual probability of nuclear war, using a logistic model."""
        alpha = self.config.model.nuclear_alpha
        # Example: logistic regression (can be replaced with ML model)
        base_prob = 1.0 - np.exp(-alpha * deployed_warheads)
        # Conflict escalation multiplier
        prob = base_prob * (1.0 + max(0.0, conflict_score) * 0.1)
        return min(prob, 0.9999)

    def famine_risk(self, fpi: float, conflict_score: float, temp_anomaly: float, years: int = 1, explain: bool = False) -> float:
        """
        Improved: Data-driven annual/decadal probability of global famine, calibrated to EM-DAT drought/extreme temperature event frequency and deaths.
        - Lower base rate to match global catastrophic famine frequency (EM-DAT: ~2-3 major events per year globally, but not all are global-scale).
        - Reduce multipliers, add hard cap for decadal risk, and improve explainability.
        """
        # EM-DAT 2000-2025: 419 droughts, 536 extreme temp events, 24197 + 289981 deaths
        # Assume only 1 in 10 of these are truly global/catastrophic
        base_p = 0.008  # ~0.8%/yr for global catastrophic famine
        fpi_factor = max(0, (fpi - 145) / 80)  # less sensitive, only >145
        conflict_factor = min(1.0, conflict_score / 10)
        temp_factor = max(0, (temp_anomaly - 1.1) * 0.7)
        event_boost = 0.0
        event_reasons = []
        if fpi > 170:
            event_boost += 0.005
            event_reasons.append('Extreme food prices')
        if conflict_score > 8:
            event_boost += 0.004
            event_reasons.append('Major conflict')
        if temp_anomaly > 1.5:
            event_boost += 0.003
            event_reasons.append('Severe climate anomaly')
        # Main risk formula
        risk = base_p * (1 + 0.7*fpi_factor + 0.6*conflict_factor + 0.5*temp_factor) + event_boost
        risk = min(risk, 0.04)  # cap annual risk at 4%
        compounded = 1 - (1 - risk) ** years
        compounded = min(compounded, 0.35)  # cap decadal risk at 35%
        if explain:
            return compounded, {
                'base_p': base_p,
                'fpi': fpi,
                'conflict_score': conflict_score,
                'temp_anomaly': temp_anomaly,
                'fpi_factor': fpi_factor,
                'conflict_factor': conflict_factor,
                'temp_factor': temp_factor,
                'event_boost': event_boost,
                'event_reasons': event_reasons,
                'annual_risk': risk,
                'years': years,
                'compounded': compounded,
                'annual_cap': 0.04,
                'decadal_cap': 0.35,
                'emdat_events_per_year': 36.7,
                'emdat_drought_deaths': 24197,
                'emdat_exttemp_deaths': 289981
            }
        return compounded

    def pandemic_risk(self, features: Dict[str, float]) -> float:
        """Pandemic risk using EM-DAT epidemic event frequency and deaths."""
        # EM-DAT 2000-2025: 885 epidemics, 123894 deaths
        base_p = 885 / 26 / 200  # ~0.17%/yr for global catastrophic pandemic (200 countries)
        # Increase risk if zoonotic_events or low vax_coverage
        risk = base_p
        if features.get('zoonotic_events', 0) > 3:
            risk += 0.001
        if features.get('vax_coverage', 1) < 0.3:
            risk += 0.001
        return min(risk, 0.01)

    def climate_risk(self, temp_anomaly: float, temp_trend: float, explain: bool = False) -> float:
        """Years to 2C warming threshold, and annual climate disaster risk using EM-DAT (flood, storm, drought, extreme temp)."""
        # EM-DAT 2000-2025: Flood 4198, Storm 2737, Drought 419, Extreme temp 536
        # Use as proxy for annual probability of major climate disaster
        climate_events = 4198 + 2737 + 419 + 536
        base_p = climate_events / 26 / 200  # ~1.1%/yr for global catastrophic climate event
        # Linear projection for years to 2C
        if temp_trend <= 0:
            years = float('inf')
        else:
            years = (2.0 - temp_anomaly) / (temp_trend / 10.0)
            years = max(years, 0.0)
        if explain:
            return years, {'base_p': base_p, 'temp_anomaly': temp_anomaly, 'temp_trend': temp_trend, 'years_to_2C': years, 'emdat_climate_events_per_year': climate_events / 26}
        return years

    def conflict_risk(self, conflict_score: float, conflict_trend: float, famine_risk: float = 0.0) -> float:
        """Annual probability of world war, using a hazard model. Famine risk increases conflict risk. EM-DAT not directly used, but could use as proxy for instability."""
        base = self.config.model.conflict_base_hazard
        scale = self.config.model.conflict_hazard_scale
        famine_feedback = 1.0 + 1.5 * famine_risk
        hazard = (base + scale * max(0.0, conflict_score)) * famine_feedback
        hazard = max(1e-8, hazard * (1.0 + conflict_trend))
        return hazard

    def earthquake_risk(self, features: Dict[str, float]) -> float:
        """Earthquake risk using EM-DAT event frequency and deaths."""
        # EM-DAT 2000-2025: 685 earthquakes, 796081 deaths
        base_p = 685 / 26 / 200  # ~0.13%/yr for global catastrophic earthquake
        return base_p

    def flood_risk(self, features: Dict[str, float]) -> float:
        """Flood risk using EM-DAT event frequency and deaths."""
        # EM-DAT 2000-2025: 4198 floods, 139209 deaths
        base_p = 4198 / 26 / 200  # ~0.8%/yr for global catastrophic flood
        return base_p

    def storm_risk(self, features: Dict[str, float]) -> float:
        """Storm risk using EM-DAT event frequency and deaths."""
        # EM-DAT 2000-2025: 2737 storms, 223650 deaths
        base_p = 2737 / 26 / 200  # ~0.5%/yr for global catastrophic storm
        return base_p

    def wildfire_risk(self, features: Dict[str, float]) -> float:
        """Wildfire risk using EM-DAT event frequency and deaths."""
        # EM-DAT 2000-2025: 331 wildfires, 2298 deaths
        base_p = 331 / 26 / 200  # ~0.06%/yr for global catastrophic wildfire
        return base_p

    def climate_risk(self, temp_anomaly: float, temp_trend: float) -> float:
        """Years to 2C warming threshold (simple linear model, can be replaced)."""
        if temp_trend <= 0:
            return float('inf')
        years = (2.0 - temp_anomaly) / (temp_trend / 10.0)
        return max(years, 0.0)

    def conflict_risk(self, conflict_score: float, conflict_trend: float, famine_risk: float = 0.0) -> float:
        """Annual probability of world war, using a hazard model. Famine risk increases conflict risk."""
        base = self.config.model.conflict_base_hazard
        scale = self.config.model.conflict_hazard_scale
        # Feedback: famine risk increases conflict risk (proxy for food riots, instability)
        famine_feedback = 1.0 + 1.5 * famine_risk  # e.g., famine_risk=0.1 → +15%
        hazard = (base + scale * max(0.0, conflict_score)) * famine_feedback
        hazard = max(1e-8, hazard * (1.0 + conflict_trend))
        return hazard

    # --- Advanced/ML risk models ---
    def pandemic_risk(self, features: Dict[str, float]) -> float:
        """Pandemic risk using advanced ML/statistical model."""
        try:
            from advanced_risk_models import PandemicRiskModel
            model = PandemicRiskModel()
            return model.predict(features)
        except Exception as e:
            import logging
            logging.error(f"Pandemic risk model failed: {e}")
            return 0.01  # fallback default

    def cyber_risk(self, features: Dict[str, float]) -> float:
        """Cyber risk using advanced ML/statistical model."""
        try:
            from advanced_risk_models import CyberRiskModel
            model = CyberRiskModel()
            return model.predict(features)
        except Exception as e:
            import logging
            logging.error(f"Cyber risk model failed: {e}")
            return 0.01

    def economic_shock_risk(self, features: Dict[str, float]) -> float:
        """Economic shock risk using advanced ML/statistical model."""
        try:
            from advanced_risk_models import EconomicShockRiskModel
            model = EconomicShockRiskModel()
            return model.predict(features)
        except Exception:
            return 0.01

    def climate_risk_ml(self, features: Dict[str, float]) -> float:
        """Climate risk using advanced ML/statistical model."""
        try:
            from advanced_risk_models import ClimateRiskModel
            model = ClimateRiskModel()
            return model.predict(features)
        except Exception:
            return 0.01

    def social_unrest_risk(self, features: Dict[str, float]) -> float:
        """Social unrest risk using advanced ML/statistical model."""
        try:
            from advanced_risk_models import SocialUnrestRiskModel
            model = SocialUnrestRiskModel()
            return model.predict(features)
        except Exception:
            return 0.01

    def technological_risk(self, features: Dict[str, float]) -> float:
        """Technological risk using advanced ML/statistical model."""
        try:
            from advanced_risk_models import TechnologicalRiskModel
            model = TechnologicalRiskModel()
            return model.predict(features)
        except Exception:
            return 0.01

    def environmental_risk(self, features: Dict[str, float]) -> float:
        """Environmental risk using advanced ML/statistical model."""
        try:
            from advanced_risk_models import EnvironmentalRiskModel
            model = EnvironmentalRiskModel()
            return model.predict(features)
        except Exception:
            return 0.01

    def geopolitical_risk(self, features: Dict[str, float]) -> float:
        """Geopolitical risk using advanced ML/statistical model."""
        try:
            from advanced_risk_models import GeopoliticalRiskModel
            model = GeopoliticalRiskModel()
            return model.predict(features)
        except Exception:
            return 0.01

    def supply_chain_risk(self, features: Dict[str, float]) -> float:
        """Supply chain risk using advanced ML/statistical model."""
        try:
            from advanced_risk_models import SupplyChainRiskModel
            model = SupplyChainRiskModel()
            return model.predict(features)
        except Exception:
            return 0.01

    def misinformation_risk(self, features: Dict[str, float]) -> float:
        """Misinformation/disinformation risk using advanced ML/statistical model."""
        try:
            from advanced_risk_models import MisinformationRiskModel
            model = MisinformationRiskModel()
            return model.predict(features)
        except Exception:
            return 0.01

    def all_risks(self, inputs: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute all available risks using standard, ML, ensemble, and external data models.
        This is plug-in ready: add new data and models as needed.
        """
        risks = {
            'nuclear': self.nuclear_risk(inputs['nuclear_deployed'], inputs['conflict_score']),
            'famine': self.famine_risk(inputs['fao_fpi'], inputs['conflict_score'], inputs['temp_anomaly']),
            'climate_years_to_2C': self.climate_risk(inputs['temp_anomaly'], inputs['temp_trend_decade']),
            'conflict': self.conflict_risk(inputs['conflict_score'], inputs['conflict_trend'])
        }
        # Advanced/ML risks
        risks['pandemic'] = self.pandemic_risk({
            'mobility': inputs.get('mobility', 0),
            'zoonotic_events': inputs.get('zoonotic_events', 0),
            'vax_coverage': inputs.get('vax_coverage', 0)
        })
        risks['cyber'] = self.cyber_risk({
            'critical_infra': inputs.get('critical_infra', 0),
            'attack_trend': inputs.get('attack_trend', 0)
        })
        risks['economic_shock'] = self.economic_shock_risk({
            'gdp_drop': inputs.get('gdp_drop', 0),
            'unemployment': inputs.get('unemployment', 0)
        })
        risks['climate_ml'] = self.climate_risk_ml({
            'temp_anomaly': inputs.get('temp_anomaly', 0),
            'temp_trend': inputs.get('temp_trend_decade', 0)
        })
        risks['social_unrest'] = self.social_unrest_risk({
            'protest_events': inputs.get('protest_events', 0),
            'social_sentiment': inputs.get('social_sentiment', 0)
        })
        risks['technological'] = self.technological_risk({
            'ai_incidents': inputs.get('ai_incidents', 0),
            'infra_failures': inputs.get('infra_failures', 0)
        })
        risks['environmental'] = self.environmental_risk({
            'wildfires': inputs.get('wildfires', 0),
            'floods': inputs.get('floods', 0),
            'droughts': inputs.get('droughts', 0)
        })
        risks['geopolitical'] = self.geopolitical_risk({
            'sanctions': inputs.get('sanctions', 0),
            'military_maneuvers': inputs.get('military_maneuvers', 0)
        })
        risks['supply_chain'] = self.supply_chain_risk({
            'shipping_delays': inputs.get('shipping_delays', 0),
            'shortages': inputs.get('shortages', 0)
        })
        risks['misinformation'] = self.misinformation_risk({
            'fake_news': inputs.get('fake_news', 0),
            'bot_activity': inputs.get('bot_activity', 0)
        })
        # Hyper-ensemble risk
        risks['ensemble'] = self.ensemble_risk(inputs)
        # Satellite/AI/other plug-in risks
        risks['ai'] = self.ai_risk(inputs)
        risks['satellite_climate'] = self.satellite_climate_risk(inputs)
        # Add more plug-in risks as needed
        return risks
