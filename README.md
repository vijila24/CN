# Dynamic Bandwidth Allocation for Online Learning (Python + Streamlit)

This mini project simulates how to allocate a shared network capacity among multiple student video sessions using different algorithms.

- Algorithms: Equal, Weighted, Proportional Fair, Min-Guarantee.
- Video model: simple ABR that chooses bitrate rung based on an EMA of recent throughput with a safety factor.
- Metrics: throughput, Jain's fairness, stall time, QoE.
- UI: Streamlit app with interactive controls and plots.

## Quick start

1. Create and activate a Python 3.10+ environment.
2. Install dependencies:
   
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   
   ```bash
   streamlit run app.py
   ```

4. Open the local URL shown in the terminal.

## How it works

- Each second, the simulator:
  - Computes per-user demand = current bitrate.
  - Allocates capacity using the selected strategy.
  - Updates per-user throughput estimate (EMA) and selects a new bitrate rung with safety margin.
  - Updates buffer and counts stall seconds when buffer hits 0.
- Summary metrics and time-series plots are shown.

## Notes

- You can mark a fraction of users as VIPs by increasing their weight and/or giving them a minimum guaranteed rate.
- Capacity varies sinusoidally around a mean with additive noise; adjust in the sidebar.

## Files

- `allocators.py`: allocation algorithms and dispatcher.
- `simulator.py`: time-stepped simulation engine and metrics.
- `app.py`: Streamlit UI.
- `requirements.txt`: dependencies.

## Educational pointers

- Jain's fairness index: J = (sum x)^2 / (n * sum x^2). Values closer to 1 are fairer.
- Proportional fairness approximated with water-filling under demand caps.
- QoE is a toy model combining bitrate, stalls, and bitrate switches.
