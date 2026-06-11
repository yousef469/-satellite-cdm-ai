"""
Production-grade constellation-scale collision avoidance decision algorithm.

Deterministic rules-based engine (no training) that sits on top of the AI risk
model. Handles: alert levels, temporal consistency, fuel budgets, maneuver
scheduling, cross-fleet prioritization, and threshold calibration.

Usage (production):
    from algorithm import ConstellationDecisionEngine, AlgorithmConfig

    engine = ConstellationDecisionEngine(config)
    decisions = engine.batch_process(satellites, conjunctions)
    for d in decisions:
        print(d.satellite_id, d.alert_level, d.maneuver)

    # Calibrate thresholds from historical data
    engine.calibrate_thresholds(risk_scores, labels, target_fpr=0.05)

    # Serialize decisions
    engine.save_decisions(decisions, "maneuver_plan.json")
"""
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────────────

@dataclass
class AlertThresholds:
    amber: float = 0.30
    red: float = 0.60
    critical: float = 0.90

    def get_level(self, risk_score: float) -> str:
        if risk_score >= self.critical:
            return "CRITICAL"
        if risk_score >= self.red:
            return "RED"
        if risk_score >= self.amber:
            return "AMBER"
        return "GREEN"


@dataclass
class AlgorithmConfig:
    thresholds: AlertThresholds = field(default_factory=AlertThresholds)
    temporal_window: int = 5
    temporal_min_positive: int = 3
    fuel_reserve_fraction: float = 0.15
    min_lead_time_hours: float = 6.0
    max_lead_time_hours: float = 48.0
    maneuver_duration_hours: float = 0.5
    # If a conjunction has no satellite state, these are the defaults
    default_fuel_kg: float = 5.0
    default_fuel_per_maneuver_kg: float = 0.15


# ── Data Structures ────────────────────────────────────────────────────────

@dataclass
class ConjunctionDecision:
    satellite_id: str
    conjunction_id: str
    risk_score: float
    alert_level: str
    time_to_tca_hours: float
    miss_distance_km: float
    probability_of_collision: float
    maneuver: bool
    confidence: float
    reason: str = ""
    schedule: Optional[dict] = None


@dataclass
class SatelliteState:
    satellite_id: str
    fuel_remaining_kg: float
    fuel_per_maneuver_kg: float
    battery_soc: float
    is_maneuvering: bool = False
    maneuvers_remaining: int = 0
    recent_scores: list = field(default_factory=list)


# ── Algorithm Components ───────────────────────────────────────────────────

class TemporalConsistencyFilter:
    """Require M out of last N risk scores above threshold before acting."""
    def __init__(self, window: int = 5, min_positive: int = 3):
        self.window = window
        self.min_positive = min_positive
        logger.debug(f"TemporalConsistencyFilter: window={window}, min_positive={min_positive}")

    def should_maneuver(self, recent_scores: list, threshold: float) -> bool:
        if len(recent_scores) < self.window:
            return False
        recent = recent_scores[-self.window:]
        above = sum(1 for s in recent if s >= threshold)
        return above >= self.min_positive


class FuelBudgetTracker:
    def __init__(self, reserve_fraction: float = 0.15):
        self.reserve_fraction = reserve_fraction
        logger.debug(f"FuelBudgetTracker: reserve_fraction={reserve_fraction}")

    def can_maneuver(self, sat: SatelliteState, alert_level: str) -> bool:
        if alert_level == "CRITICAL":
            return True
        fuel_after = sat.fuel_remaining_kg - sat.fuel_per_maneuver_kg
        min_fuel = sat.fuel_per_maneuver_kg * self.reserve_fraction * max(sat.maneuvers_remaining, 1)
        decision = fuel_after >= min_fuel
        if not decision:
            logger.info(f"  Fuel block: {sat.satellite_id} has {sat.fuel_remaining_kg:.2f}kg, needs {min_fuel:.2f}kg reserve")
        return decision


class ManeuverScheduler:
    def __init__(self, min_lead_hours: float = 6.0, max_lead_hours: float = 48.0,
                 duration_hours: float = 0.5):
        self.min_lead = min_lead_hours
        self.max_lead = max_lead_hours
        self.duration = duration_hours

    def schedule(self, time_to_tca_hours: float) -> Optional[dict]:
        if time_to_tca_hours < self.min_lead:
            return None
        window_end = min(time_to_tca_hours, self.max_lead)
        scheduled = max(self.min_lead, window_end * 0.3)
        return {
            'start_hours_before_tca': round(scheduled, 2),
            'window_open': round(scheduled, 2),
            'window_close': round(window_end, 2),
            'duration_hours': self.duration,
        }


# ── Urgency Scorer ─────────────────────────────────────────────────────────

def compute_urgency(decision: ConjunctionDecision, sat: SatelliteState) -> float:
    urgency = 0.0
    if decision.alert_level == "CRITICAL":
        urgency += 100.0
    elif decision.alert_level == "RED":
        urgency += 50.0

    urgency += (1.0 - min(decision.time_to_tca_hours, 168.0) / 168.0) * 30.0
    urgency += decision.risk_score * 20.0
    urgency += (1.0 - min(decision.miss_distance_km, 10.0) / 10.0) * 10.0

    if sat.fuel_remaining_kg < sat.fuel_per_maneuver_kg * 3:
        urgency -= 40.0

    if decision.time_to_tca_hours < 12:
        urgency += 25.0

    return max(0.0, urgency)


# ── Main Engine ────────────────────────────────────────────────────────────

class ConstellationDecisionEngine:
    """
    Production decision engine for constellation-scale CA.

    Pipeline per conjunction:
      1. Compute alert level from risk score
      2. Check temporal consistency (last N CDMs)
      3. Check fuel budget
      4. Schedule maneuver window
      5. Rank by urgency across the fleet
      6. Return ordered decision list
    """
    def __init__(self, config: Optional[AlgorithmConfig] = None):
        cfg = config or AlgorithmConfig()
        self.thresholds = cfg.thresholds
        self.temporal = TemporalConsistencyFilter(cfg.temporal_window, cfg.temporal_min_positive)
        self.fuel = FuelBudgetTracker(cfg.fuel_reserve_fraction)
        self.scheduler = ManeuverScheduler(cfg.min_lead_time_hours, cfg.max_lead_time_hours, cfg.maneuver_duration_hours)
        self.config = cfg
        self.metrics = {'total_processed': 0, 'total_maneuvers': 0, 'avg_latency_ms': 0.0}
        logger.info(f"ConstellationDecisionEngine initialized: "
                     f"amber={cfg.thresholds.amber}, red={cfg.thresholds.red}, critical={cfg.thresholds.critical}")

    def evaluate_cdm(self, satellite: SatelliteState,
                     risk_score: float, time_to_tca_hours: float,
                     miss_distance_km: float = 10.0,
                     probability_of_collision: float = 0.0,
                     conjunction_id: str = "unknown") -> ConjunctionDecision:
        t0 = time.perf_counter()
        level = self.thresholds.get_level(risk_score)
        satellite.recent_scores.append(risk_score)

        temporal_ok = self.temporal.should_maneuver(satellite.recent_scores, self.thresholds.red)
        fuel_ok = self.fuel.can_maneuver(satellite, level)

        if level == "CRITICAL":
            maneuver = True
        elif level == "RED" and temporal_ok and fuel_ok:
            maneuver = True
        else:
            maneuver = False

        confidence = min(1.0, risk_score * (1.1 if level == "CRITICAL" else 1.0))
        if not temporal_ok or not fuel_ok:
            confidence *= 0.8

        reasons = []
        if level == "CRITICAL":
            reasons.append("Critical risk threshold exceeded")
        if not temporal_ok and level != "GREEN":
            reasons.append("Not yet confirmed temporally")
        if not fuel_ok and level != "CRITICAL":
            reasons.append("Insufficient fuel")
        if maneuver:
            reasons.append("Maneuver recommended")

        sched = self.scheduler.schedule(time_to_tca_hours) if maneuver else None
        latency = (time.perf_counter() - t0) * 1000

        self.metrics['total_processed'] += 1
        if maneuver:
            self.metrics['total_maneuvers'] += 1
        self.metrics['avg_latency_ms'] += (latency - self.metrics['avg_latency_ms']) / max(self.metrics['total_processed'], 1)

        return ConjunctionDecision(
            satellite_id=satellite.satellite_id,
            conjunction_id=conjunction_id,
            risk_score=round(risk_score, 4),
            alert_level=level,
            time_to_tca_hours=round(time_to_tca_hours, 2),
            miss_distance_km=round(miss_distance_km, 3),
            probability_of_collision=round(probability_of_collision, 6),
            maneuver=maneuver,
            confidence=round(confidence, 3),
            reason=" | ".join(reasons) if reasons else "No action needed",
            schedule=sched,
        )

    def batch_process(self, satellites: dict[str, SatelliteState],
                      conjunctions: list[dict]) -> list[ConjunctionDecision]:
        if not conjunctions:
            logger.warning("batch_process called with empty conjunction list")
            return []

        logger.info(f"Batch processing {len(conjunctions)} conjunctions across {len(satellites)} satellites")
        decisions = []
        errors = 0

        for conj in conjunctions:
            try:
                sat = satellites.get(conj['satellite_id'])
                if sat is None:
                    sat = SatelliteState(
                        satellite_id=conj['satellite_id'],
                        fuel_remaining_kg=self.config.default_fuel_kg,
                        fuel_per_maneuver_kg=self.config.default_fuel_per_maneuver_kg,
                        battery_soc=1.0,
                    )
                    satellites[conj['satellite_id']] = sat

                dec = self.evaluate_cdm(
                    satellite=sat,
                    risk_score=conj['risk_score'],
                    time_to_tca_hours=conj['time_to_tca_hours'],
                    miss_distance_km=conj.get('miss_distance_km', 10.0),
                    probability_of_collision=conj.get('probability_of_collision', 0.0),
                    conjunction_id=conj.get('conjunction_id', 'unknown'),
                )
                decisions.append(dec)
            except Exception as e:
                errors += 1
                logger.error(f"Error processing conjunction {conj.get('conjunction_id', 'unknown')}: {e}")

        decisions.sort(key=lambda d: compute_urgency(
            d, satellites.get(d.satellite_id, SatelliteState(d.satellite_id, 0, 0, 0))
        ), reverse=True)

        logger.info(f"  Processed: {len(decisions)}, errors: {errors}, maneuvers: {self.metrics['total_maneuvers']}")
        return decisions

    def calibrate_thresholds(self, risk_scores: list[float], labels: list[int],
                              target_fpr: float = 0.05) -> AlertThresholds:
        if not risk_scores or not labels:
            logger.warning("calibrate_thresholds called with empty data")
            return self.thresholds

        scores = np.array(risk_scores)
        lbls = np.array(labels)
        neg_scores = scores[lbls == 0]

        if len(neg_scores) < 10:
            logger.warning(f"Too few negative samples ({len(neg_scores)}) for calibration")
            return self.thresholds

        idx = max(0, min(int(len(neg_scores) * (1.0 - target_fpr)), len(neg_scores) - 1))
        red_th = float(np.sort(neg_scores)[idx])

        pos_scores = scores[lbls == 1]
        amber_th = red_th * 0.5 if len(pos_scores) > 0 else self.thresholds.amber
        critical_th = red_th * 1.5 if len(pos_scores) > 0 else self.thresholds.critical

        self.thresholds = AlertThresholds(
            amber=max(0.01, amber_th),
            red=max(0.05, red_th),
            critical=min(0.99, max(0.10, critical_th)),
        )
        logger.info(f"Calibrated thresholds: amber={self.thresholds.amber:.3f}, "
                     f"red={self.thresholds.red:.3f}, critical={self.thresholds.critical:.3f}")
        return self.thresholds

    # ── Serialization ─────────────────────────────────────────────────────

    @staticmethod
    def decision_to_dict(d: ConjunctionDecision) -> dict:
        return asdict(d)

    @staticmethod
    def decisions_to_dicts(decisions: list[ConjunctionDecision]) -> list[dict]:
        return [asdict(d) for d in decisions]

    def save_decisions(self, decisions: list[ConjunctionDecision], path: str):
        data = {
            'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'engine_metrics': self.metrics,
            'thresholds': asdict(self.thresholds),
            'decisions': self.decisions_to_dicts(decisions),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved {len(decisions)} decisions to {path}")

    def get_metrics(self) -> dict:
        return dict(self.metrics)


def generate_decision_report(decisions: list[ConjunctionDecision]) -> dict:
    if not decisions:
        return {'total_conjunctions': 0, 'maneuvers_recommended': 0}

    maneuvers = [d for d in decisions if d.maneuver]
    critical = [d for d in decisions if d.alert_level == "CRITICAL"]
    red = [d for d in decisions if d.alert_level == "RED"]
    amber = [d for d in decisions if d.alert_level == "AMBER"]
    by_sat = set(d.satellite_id for d in decisions)
    risk_scores = [d.risk_score for d in decisions]

    return {
        'total_conjunctions': len(decisions),
        'maneuvers_recommended': len(maneuvers),
        'critical_alerts': len(critical),
        'red_alerts': len(red),
        'amber_alerts': len(amber),
        'green': len(decisions) - len(critical) - len(red) - len(amber),
        'satellites_affected': len(by_sat),
        'avg_risk_score': float(np.mean(risk_scores)),
        'max_risk_score': float(max(risk_scores)),
    }
