import streamlit as st
import numpy as np
import pandas as pd

from simulator import SimConfig, run_simulation, default_ladder
from allocators import STRATEGY_NAMES

st.set_page_config(page_title="Dynamic Bandwidth Allocation Simulator", layout="wide")

st.title("Dynamic Bandwidth Allocation for Online Learning Platforms")

with st.sidebar:
    st.header("Simulation Controls")
    users = st.slider("Users", 2, 50, 20, 1)
    seconds = st.slider("Duration (s)", 60, 1200, 600, 30)
    cap_mean = st.slider("Capacity mean (Mbps)", 2.0, 100.0, 20.0, 0.5)
    cap_var = st.slider("Capacity variability (fraction)", 0.0, 1.0, 0.3, 0.05)
    noise = st.slider("Noise std (Mbps)", 0.0, 10.0, 1.0, 0.1)

    strategy = st.selectbox("Allocation Strategy", STRATEGY_NAMES, index=0)

    vip_fraction = st.slider("VIP fraction (Weighted/Min-Guarantee)", 0.0, 1.0, 0.2, 0.05)
    vip_weight = st.slider("VIP weight", 1.0, 10.0, 3.0, 0.5)
    min_rate = st.slider("Min rate for VIPs (Mbps)", 0.0, 10.0, 1.0, 0.1)

    alpha = st.slider("ABR safety factor (alpha)", 0.5, 1.0, 0.85, 0.01)
    ema = st.slider("Throughput EMA", 0.1, 0.95, 0.7, 0.05)
    init_buf = st.slider("Initial buffer (s)", 0.0, 30.0, 10.0, 0.5)
    max_buf = st.slider("Max buffer (s)", 1.0, 60.0, 30.0, 1.0)

    stall_pen = st.slider("Stall penalty", 0.0, 10.0, 4.0, 0.5)
    switch_pen = st.slider("Switch penalty", 0.0, 5.0, 0.5, 0.1)

    seed = st.number_input("Random seed", 0, 10_000, 42, 1)

    run_btn = st.button("Run Simulation", type="primary")

# Build config
cfg = SimConfig(
    users=users,
    seconds=seconds,
    capacity_mean=cap_mean,
    capacity_var=cap_var,
    noise_std=noise,
    alpha_throughput=alpha,
    ema=ema,
    init_buffer=init_buf,
    max_buffer=max_buf,
    stall_penalty=stall_pen,
    switch_penalty=switch_pen,
    seed=int(seed),
)

n = users
vip_count = int(round(vip_fraction * n))
weights = np.ones(n)
if vip_count > 0:
    weights[:vip_count] = vip_weight
min_rates = np.zeros(n)
if vip_count > 0:
    min_rates[:vip_count] = min_rate

if run_btn:
    with st.spinner("Simulating..."):
        out = run_simulation(strategy, cfg, weights=weights, min_rates=min_rates, bitrate_ladder=default_ladder())

    st.subheader("Overall Metrics")
    cols = st.columns(5)
    cols[0].metric("Mean capacity (Mbps)", f"{out['overall']['mean_capacity_mbps']:.2f}")
    cols[1].metric("Mean alloc (Mbps)", f"{out['overall']['mean_alloc_mbps']:.2f}")
    cols[2].metric("Jain fairness", f"{out['overall']['fairness_jain']:.3f}")
    cols[3].metric("Avg stall (s)", f"{out['overall']['avg_stall_seconds']:.2f}")
    cols[4].metric("Avg QoE", f"{out['overall']['avg_qoe']:.2f}")

    st.subheader("Time Series")
    cap_series = pd.DataFrame({"time": np.arange(cfg.seconds), "capacity": out["capacity"]})
    st.line_chart(cap_series.set_index("time"))

    st.write("Allocations (subset for clarity)")
    df_a = out["df_alloc"].copy()
    df_a_small = df_a[["time"] + [c for c in df_a.columns if c != "time"][:min(8, n)]]
    st.line_chart(df_a_small.set_index("time"))

    st.write("Bitrates (subset for clarity)")
    df_b = out["df_bitrate"].copy()
    df_b_small = df_b[["time"] + [c for c in df_b.columns if c != "time"][:min(8, n)]]
    st.line_chart(df_b_small.set_index("time"))

    st.write("Buffers (subset for clarity)")
    df_buf = out["df_buffer"].copy()
    df_buf_small = df_buf[["time"] + [c for c in df_buf.columns if c != "time"][:min(8, n)]]
    st.line_chart(df_buf_small.set_index("time"))

    st.subheader("Per-user Summary")
    st.dataframe(out["summary"].style.format({
        "avg_alloc_mbps": "{:.2f}",
        "avg_bitrate_mbps": "{:.2f}",
        "stall_seconds": "{:.0f}",
        "qoe": "{:.2f}",
        "weight": "{:.1f}",
        "min_rate": "{:.1f}",
    }), use_container_width=True)

else:
    st.info("Set parameters in the sidebar and click 'Run Simulation'.")
