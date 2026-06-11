"""
Constellation-Scale Collision Avoidance Decision Algorithm

Inspired by SpaceX Problem #3: automated decision-making for 1000s of satellites.

Pipeline:
  1. AI Model → risk score (0-1) per conjunction
  2. Algorithm → maneuver/no-maneuver per conjunction, ranked by urgency

This is a deterministic, rules-based algorithm — no training required.
It encodes operational constraints that make CA practical at scale:
fuel budgets, false alarm tolerance, maneuver windows, priority ordering.

Usage:
    from algorithm import ConstellationDecisionEngine, AlertThresholds

    engine = ConstellationDecisionEngine()
    decisions = engine.batch_process(satellites, conjunctions)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────
#  Alert Levels
# ──────────────────────────────────────────────
@dataclass
class AlertThresholds:
    amber: float = 0.30   # Monitor — above this, start watching
    red: float = 0.60     # Maneuver — above this, plan a maneuver
    critical: float = 0.90  # Immediate — maneuver ASAP regardless of fuel

    def get_level(self, risk_score: float) -> str:
        if risk_score >= self.critical:
            return "CRITICAL"
        elif risk_score >= self.red:
            return "RED"
        elif risk_score >= self.amber:
            return "AMBER"
        return "GREEN"


# ──────────────────────────────────────────────
#  Conjunction Decision
# ──────────────────────────────────────────────
@dataclass
class ConjunctionDecision:
    """Decision for a single conjunction event on one satellite."""
    satellite_id: str
    conjunction_id: str
    risk_score: float
    alert_level: str
    time_to_tca_hours: float
    miss_distance_km: float
    probability_of_collision: float
    maneuver: bool
    confidence: float  # how confident in this decision (0-1)
    reason: str = ""


@dataclass
class SatelliteState:
    """Current operational state of one satellite."""
    satellite_id: str
    fuel_remaining_kg: float
    fuel_per_maneuver_kg: float  # typical fuel cost per maneuver
    battery_soc: float  # state of charge 0-1
    is_maneuvering: bool = False
    maneuvers_remaining: int = 0
    recent_scores: list = field(default_factory=list)  # last N risk scores for temporal consistency


# ──────────────────────────────────────────────
#  Temporal Consistency Filter
# ──────────────────────────────────────────────
class TemporalConsistencyFilter:
    """
    Require M out of last N risk scores above threshold before acting.
    This prevents maneuver decisions from a single spurious high score.
    """
    def __init__(self, window: int = 5, min_positive: int = 3):
        self.window = window
        self.min_positive = min_positive

    def should_maneuver(self, recent_scores: list, threshold: float) -> bool:
        if len(recent_scores) < self.window:
            return False  # not enough data yet
        recent = recent_scores[-self.window:]
        above = sum(1 for s in recent if s >= threshold)
        return above >= self.min_positive


# ──────────────────────────────────────────────
#  Fuel Budget Tracker
# ──────────────────────────────────────────────
class FuelBudgetTracker:
    """
    Tracks fuel across the constellation and prevents maneuvers
    when fuel is too low (unless critical).
    """
    def __init__(self, reserve_fraction: float = 0.15):
        self.reserve_fraction = reserve_fraction

    def can_maneuver(self, sat: SatelliteState, alert_level: str) -> bool:
        if alert_level == "CRITICAL":
            return True  # always maneuver for critical
        fuel_after = sat.fuel_remaining_kg - sat.fuel_per_maneuver_kg
        min_fuel = sat.fuel_per_maneuver_kg * self.reserve_fraction * sat.maneuvers_remaining
        return fuel_after >= min_fuel


# ──────────────────────────────────────────────
#  Maneuver Scheduler
# ──────────────────────────────────────────────
class ManeuverScheduler:
    """
    Determines when to execute a maneuver based on TCA.
    Returns earliest safe maneuver window.
    """
    def __init__(self, min_lead_time_hours: float = 6.0,
                 max_lead_time_hours: float = 48.0,
                 maneuver_duration_hours: float = 0.5):
        self.min_lead = min_lead_time_hours
        self.max_lead = max_lead_time_hours
        self.duration = maneuver_duration_hours

    def schedule(self, time_to_tca_hours: float) -> Optional[dict]:
        if time_to_tca_hours < self.min_lead:
            return None  # too late to maneuver safely
        window_end = min(time_to_tca_hours, self.max_lead)
        # Schedule at the most fuel-efficient time (closer to TCA = more precise)
        # But leave enough margin
        scheduled_hours_before = max(self.min_lead, window_end * 0.3)
        return {
            'start_hours_before_tca': scheduled_hours_before,
            'window_open': scheduled_hours_before,
            'window_close': window_end,
            'duration_hours': self.duration,
        }


# ──────────────────────────────────────────────
#  Urgency Scorer — ranks satellites that need maneuvers
# ──────────────────────────────────────────────
def compute_urgency(decision: ConjunctionDecision, sat: SatelliteState) -> float:
    """
    Higher score = more urgent. Used to prioritize when multiple
    satellites need maneuvers simultaneously.
    """
    urgency = 0.0
    if decision.alert_level == "CRITICAL":
        urgency += 100.0
    elif decision.alert_level == "RED":
        urgency += 50.0

    urgency += (1.0 - decision.time_to_tca_hours / 168.0) * 30.0  # closer to TCA = more urgent
    urgency += decision.risk_score * 20.0  # higher risk = more urgent
    urgency += (1.0 - decision.miss_distance_km / 10.0) * 10.0  # smaller miss = more urgent

    if sat.fuel_remaining_kg < sat.fuel_per_maneuver_kg * 3:
        urgency -= 40.0  # low fuel = less willing to maneuver (non-critical)

    if decision.time_to_tca_hours < 12:
        urgency += 25.0  # under 12h to TCA = very urgent

    return max(0.0, urgency)


# ──────────────────────────────────────────────
#  Main Decision Engine
# ──────────────────────────────────────────────
class ConstellationDecisionEngine:
    """
    Orchestrates the full decision pipeline for an entire constellation.

    Pipeline:
      1. Get risk scores from AI model for each CDM
      2. Apply alert thresholds → alert level per conjunction
      3. Temporal consistency filter → confirm persistent threats
      4. Fuel check → only maneuver if fuel allows (unless critical)
      5. Schedule maneuver window
      6. Prioritize across constellation
      7. Output ordered list of recommended maneuvers

    This is a rules-based algorithm. No training needed.
    """
    def __init__(self, thresholds: Optional[AlertThresholds] = None,
                 temporal: Optional[TemporalConsistencyFilter] = None,
                 fuel: Optional[FuelBudgetTracker] = None,
                 scheduler: Optional[ManeuverScheduler] = None):
        self.thresholds = thresholds or AlertThresholds()
        self.temporal = temporal or TemporalConsistencyFilter()
        self.fuel = fuel or FuelBudgetTracker()
        self.scheduler = scheduler or ManeuverScheduler()

    def evaluate_cdm(self, satellite: SatelliteState,
                     risk_score: float, time_to_tca_hours: float,
                     miss_distance_km: float, probability_of_collision: float,
                     conjunction_id: str = "unknown") -> ConjunctionDecision:
        level = self.thresholds.get_level(risk_score)
        satellite.recent_scores.append(risk_score)

        temporal_ok = self.temporal.should_maneuver(
            satellite.recent_scores, self.thresholds.red
        )
        fuel_ok = self.fuel.can_maneuver(satellite, level)

        if level == "CRITICAL":
            maneuver = True
        elif level == "RED" and temporal_ok and fuel_ok:
            maneuver = True
        elif level == "RED" and not fuel_ok:
            maneuver = False
        elif level == "RED" and not temporal_ok:
            maneuver = False
        else:
            maneuver = False

        confidence = risk_score
        if level == "CRITICAL":
            confidence = min(1.0, risk_score * 1.1)
        elif temporal_ok and fuel_ok:
            confidence = min(1.0, risk_score * 1.05)
        else:
            confidence = risk_score * 0.8

        reasons = []
        if level == "CRITICAL":
            reasons.append("Critical risk threshold exceeded")
        if not temporal_ok:
            reasons.append("Not yet confirmed temporally")
        if not fuel_ok and level != "CRITICAL":
            reasons.append("Insufficient fuel for non-critical maneuver")
        if maneuver:
            reasons.append("Maneuver recommended")

        return ConjunctionDecision(
            satellite_id=satellite.satellite_id,
            conjunction_id=conjunction_id,
            risk_score=risk_score,
            alert_level=level,
            time_to_tca_hours=time_to_tca_hours,
            miss_distance_km=miss_distance_km,
            probability_of_collision=probability_of_collision,
            maneuver=maneuver,
            confidence=round(confidence, 3),
            reason=" | ".join(reasons) if reasons else "No action needed",
        )

    def batch_process(self, satellites: dict[str, SatelliteState],
                      conjunctions: list[dict]) -> list[ConjunctionDecision]:
        """
        Process all conjunctions across the constellation in one batch.

        conjunctions: list of dicts with keys:
            satellite_id, risk_score, time_to_tca_hours, miss_distance_km,
            probability_of_collision, conjunction_id
        """
        decisions = []
        for conj in conjunctions:
            sat = satellites.get(conj['satellite_id'])
            if sat is None:
                continue
            dec = self.evaluate_cdm(
                satellite=sat,
                risk_score=conj['risk_score'],
                time_to_tca_hours=conj['time_to_tca_hours'],
                miss_distance_km=conj.get('miss_distance_km', 10.0),
                probability_of_collision=conj.get('probability_of_collision', 0.0),
                conjunction_id=conj.get('conjunction_id', 'unknown'),
            )
            decisions.append(dec)

        # Sort by urgency (highest first)
        decisions.sort(
            key=lambda d: compute_urgency(d, satellites.get(d.satellite_id, SatelliteState(d.satellite_id, 0, 0, 0))),
            reverse=True,
        )
        return decisions

    def calibrate_thresholds(self, risk_scores: list[float],
                              labels: list[int],
                              target_fpr: float = 0.05) -> AlertThresholds:
        """
        Calibrate the red threshold to achieve a target false positive rate.
        This is the only step that uses labelled data — it sets thresholds
        from historical performance.

        No model training involved — just percentile-based threshold selection.
        """
        scores = np.array(risk_scores)
        lbls = np.array(labels)

        neg_scores = scores[lbls == 0]
        if len(neg_scores) == 0:
            return self.thresholds

        neg_scores_sorted = np.sort(neg_scores)
        idx = int(len(neg_scores_sorted) * (1.0 - target_fpr))
        idx = max(0, min(idx, len(neg_scores_sorted) - 1))
        red_th = neg_scores_sorted[idx]

        pos_scores = scores[lbls == 1]
        amber_th = red_th * 0.5 if len(pos_scores) > 0 else self.thresholds.amber
        critical_th = red_th * 1.5 if len(pos_scores) > 0 else self.thresholds.critical

        self.thresholds = AlertThresholds(
            amber=max(0.01, amber_th),
            red=max(0.05, red_th),
            critical=min(0.99, max(0.10, critical_th)),
        )
        return self.thresholds


# ──────────────────────────────────────────────
#  Report Generator
# ──────────────────────────────────────────────
def generate_decision_report(decisions: list[ConjunctionDecision]) -> dict:
    """Summarize the constellation-wide decision state."""
    total = len(decisions)
    maneuvers = [d for d in decisions if d.maneuver]
    critical = [d for d in decisions if d.alert_level == "CRITICAL"]
    red = [d for d in decisions if d.alert_level == "RED"]
    amber = [d for d in decisions if d.alert_level == "AMBER"]

    # Group by satellite
    by_sat = {}
    for d in decisions:
        by_sat.setdefault(d.satellite_id, []).append(d)

    return {
        'total_conjunctions': total,
        'maneuvers_recommended': len(maneuvers),
        'critical_alerts': len(critical),
        'red_alerts': len(red),
        'amber_alerts': len(amber),
        'green': total - len(critical) - len(red) - len(amber),
        'satellites_affected': len(by_sat),
        'avg_risk_score': np.mean([d.risk_score for d in decisions]) if decisions else 0.0,
        'max_risk_score': max(d.risk_score for d in decisions) if decisions else 0.0,
    }
