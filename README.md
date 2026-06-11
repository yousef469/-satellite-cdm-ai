# CDM-AI: Satellite Collision Avoidance with Temporal Deep Learning

**AI model + decision algorithm for constellation-scale conjunction risk assessment.**

This project was inspired by SpaceX's Problem #3 on automated collision avoidance
for mega-constellations. It provides an end-to-end pipeline: generate synthetic CDM
data, train a temporal neural network, and run a rules-based decision engine across
thousands of satellites.

## Why This Exists

Current collision avoidance relies on simple probability-of-collision (Pc) thresholds
from CSpOC screens. This produces high false alarm rates — wasting fuel, reducing
satellite lifespan, and overwhelming operators.

CDM-AI replaces manual thresholding with a learned model that processes the full
CDM time series (6–15 messages per conjunction), captures temporal risk evolution,
and feeds interpretable risk scores into an automated decision engine.

### Key Capabilities

| Capability | What It Does |
|------------|-------------|
| **Temporal ML** | Learns risk patterns across sequential CDMs (not just the latest Pc) |
| **Production algorithm** | Fuel-aware, temporal-consistency-filtered, priority-ranked decisions |
| **Constellation scale** | Batch processes any number of satellites in O(N) |
| **Synthetic training** | Built-in generator produces realistic CDM events for model development |
| **ESA Kelvins compatible** | Loader for the ESA Collision Avoidance Challenge dataset |
| **40K params** | Small enough to train on CPU in ~30 seconds |

## How It Works

### Architecture

```
CDM Time Series (42 features, 6-15 messages per event)
        │
        ▼
  ┌─────────────┐
  │ LayerNorm   │
  │ + TimeEnc   │
  └──────┬──────┘
         │
  ┌──────▼──────┐         ┌──────────────────┐
  │ InputProj   │         │ Risk Shortcut    │
  │ 106 → 64    │         │ Linear(42 → 1)   │
  └──────┬──────┘         │ on last CDM feat │
         │                 └────────┬─────────┘
  ┌──────▼──────┐                  │
  │ DepthwiseTCN│                  │
  │ 3 blocks    │                  │
  └──────┬──────┘                  │
         │                         │
  ┌──────▼──────┐                  │
  │ CrossAttn   │                  │
  │ (pooling)   │                  │
  └──────┬──────┘                  │
         │                         │
  ┌──────▼──────┐                  │
  │ PDM         │                  │
  │ (prototypes)│                  │
  └──────┬──────┘                  │
         │                         │
         ▼                         ▼
   Representation          Risk Score (0-1)
   (for contrastive        ────────────────►
    + orthogonality              │
    losses)                      ▼
                         ┌──────────────────┐
                         │ Decision Engine  │
                         │ • Alert levels   │
                         │ • Temporal filt  │
                         │ • Fuel budget    │
                         │ • Priority rank  │
                         └──────────────────┘
```

### Two Components

**1. AI Model** (`model.py`) — learns to score risk from temporal CDM features.
- DepthwiseTCN captures multi-scale temporal patterns across sequential CDMs
- Cross-attention pools variable-length sequences to a fixed representation
- PDM (Prototype Distribution Module) learns risk archetypes via contrastive loss
- Linear shortcut from raw last-CDM features ensures a reliable gradient path

**2. Decision Algorithm** (`algorithm.py`) — rules-based logic on top of risk scores.
- Alert thresholds: GREEN → AMBER → RED → CRITICAL
- Temporal consistency: requires M/N consecutive high scores before maneuvering
- Fuel budget: prevents non-critical maneuvers when reserves are low
- Maneuver scheduler: computes safe execution windows before TCA
- Urgency scoring: ranks conjunctions across the constellation

## Quick Start

```bash
pip install torch numpy

# Train on synthetic data (30s on CPU, ~40K param model)
python run.py --mode train

# Evaluate the trained model
python run.py --mode eval

# Run full constellation demo (300 satellites, 400+ CDMs)
python run.py --mode demo
```

### Example Output (Demo)

```
============================================================
FULL PIPELINE: AI Model + Decision Algorithm
============================================================

1. Simulating 300-satellite constellation pass...
   Total CDMs: 447
   Real risks: 19

2. Running constellation-scale decision algorithm...

3. Decision Report:
   Total conjunctions:  447
   Maneuvers ordered:   8
   Critical alerts:     8
   Red alerts:          26
   Amber alerts:        53
   Safe (green):        360

4. Top-5 Most Urgent Decisions:
   Satellite            Risk     Level      TCA(h)   Miss(km)   Maneuver
   ------------------------------------------------------------------
   STARLINK-30012       0.964    CRITICAL   1.0      1.906      YES
   STARLINK-30259       0.984    CRITICAL   0.5      3.579      YES
   STARLINK-30245       0.958    CRITICAL   0.9      4.153      YES

5. Performance:
   Maneuver savings: 98% (447 → 8)
```

## Configuration

All hyperparameters are in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `HIDDEN_DIM` | 64 | TCN hidden dimension |
| `TCN_BLOCKS` | 3 | Number of temporal conv blocks |
| `NUM_PROTOTYPES` | 16 | PDM archetype count |
| `EPOCHS` | 30 | Training epochs |
| `RISK_RATIO` | 0.04 | Fraction of risky events in synthetic data |

## File Structure

```
satellite-cdm-ai/
├── algorithm.py      # Constellation decision engine (rules-based, no training)
├── cdm_generator.py  # Synthetic CDM data generator
├── config.py         # Hyperparameters and settings
├── esa_loader.py     # ESA Kelvins dataset loader
├── evaluate.py       # Precision/recall, maneuver savings
├── model.py          # PDM + DepthwiseTCN model
├── run.py            # CLI: train / eval / demo
├── train.py          # Training loop and collation
└── README.md
```

## Data Requirements for Production

The synthetic generator is for development. To achieve production recall (>90% at low FPR):

| Data | What | Minimum Count |
|------|------|--------------|
| Historical CDMs | Real conjunction messages from CSpOC/SSA | 50,000+ |
| Labeled outcomes | Which required a maneuver (ground truth) | All of above |
| Satellite telemetry | Fuel levels, battery, maneuver history | Parallel data |

The model architecture is designed to train on real data with minimal changes — the
risk shortcut learns the true Pc/maneuver relationship, while the PDM backbone
captures temporal risk archetypes.

## License

MIT — see [LICENSE](LICENSE). Use freely, but the authors assume no liability
for collision decisions made with this software.

## Acknowledgments

Inspired by SpaceX's Problem #3 on automated constellation-scale collision
avoidance, and the ESA Kelviss Collision Avoidance Challenge.
