import streamlit as st
import scipy.optimize as optimize
from itertools import combinations
from math import prod
import pandas as pd
import math

# ── Helper: convert gross return (stake included) → net return ────────────────
def gross_to_net(gross):
    return float(gross) - 1.0

# ── Helper: implied probability from American odds ───────────────────────────
def american_to_prob(odds):
    odds = float(odds)
    if odds < 0:
        p = -odds / (-odds + 100.0)
    else:
        p = 100.0 / (odds + 100.0)
    return min(max(p, 1e-12), 1-1e-12)

# ── Helper: auto-detect odds or probability input ────────────────────────────
def parse_odds_input(x_raw):
    """
    Accepts inputs like -110, +150, 55, or 55%.
    - American odds if x <= -100 or x >= 100
    - Probability if 0 <= x < 100 (percentage)
    """
    if isinstance(x_raw, str) and x_raw.strip().endswith("%"):
        val = float(x_raw.strip().strip("%"))
        return max(min(val / 100.0, 1 - 1e-12), 1e-12)
    try:
        val = float(x_raw)
    except:
        return 0.5
    if val <= -100 or val >= 100:
        return american_to_prob(val)
    elif 0 <= val < 100:
        return max(min(val / 100.0, 1 - 1e-12), 1e-12)
    else:
        return 0.5

# ── Helper: robust root‐finder for Kelly f ───────────────────────────────────
def _solve_kelly(kelly_eq):
    try:
        if kelly_eq(0.0) <= 0:
            return 0.0
    except:
        return 0.0
    try:
        sol = optimize.root_scalar(kelly_eq, bracket=[0.0, 0.9999], method="brentq")
        if sol.converged and 0 <= sol.root < 1:
            return sol.root
    except:
        pass
    return 0.0

# ── 4-leg Kelly ──────────────────────────────────────────────────────────────
def calculate_4_leg_kelly(probabilities, mults, net4, net3):
    p = probabilities
    q = [1 - x for x in p]

    P4 = prod(p)
    P3 = [
        p[0] * p[1] * p[2] * q[3],
        p[0] * p[1] * q[2] * p[3],
        p[0] * q[1] * p[2] * p[3],
        q[0] * p[1] * p[2] * p[3],
    ]
    P_L = max(1 - (P4 + sum(P3)), 0.0)

    b4 = net4 * prod(mults)
    b3 = [
        net3 * mults[0] * mults[1] * mults[2],
        net3 * mults[0] * mults[1] * mults[3],
        net3 * mults[0] * mults[2] * mults[3],
        net3 * mults[1] * mults[2] * mults[3],
    ]

    def eq(f):
        if not (0 <= f < 1):
            return float("inf")
        win_sum = (P4 * b4) / (1 + f * b4)
        for Pi, bi in zip(P3, b3):
            win_sum += (Pi * bi) / (1 + f * bi)
        return win_sum - (P_L / (1 - f))

    f_star = _solve_kelly(eq)
    ctx = {"P4": P4, "P3": P3, "P_L": P_L, "b4": b4, "b3": b3}
    return f_star, ctx

# ── 5-leg Kelly ──────────────────────────────────────────────────────────────
def calculate_5_leg_kelly(probabilities, nets, mults):
    p = probabilities
    q = [1 - x for x in p]

    # --- Accurate Kelly Calculation (Iterative) ---
    # 1. 5 of 5 (1 outcome)
    P5 = prod(p)
    b5 = nets[0] * prod(mults)

    # 2. 4 of 5 (5 outcomes)
    P4_list, b4_list = [], []
    for combo in combinations(range(5), 4):
        miss_idx = (set(range(5)) - set(combo)).pop()
        prob = prod(p[i] for i in combo) * q[miss_idx]
        payout_mult = prod(mults[i] for i in combo)
        
        P4_list.append(prob)
        b4_list.append(nets[1] * payout_mult)

    # 3. 3 of 5 (10 outcomes)
    P3_list, b3_list = [], []
    for combo in combinations(range(5), 3):
        miss_indices = set(range(5)) - set(combo)
        prob = prod(p[i] for i in combo) * prod(q[j] for j in miss_indices)
        payout_mult = prod(mults[i] for i in combo)
        
        P3_list.append(prob)
        b3_list.append(nets[2] * payout_mult)
    
    # 4. Total probabilities for loss and context
    P4_total = sum(P4_list)
    P3_total = sum(P3_list)
    P_L = max(1 - (P5 + P4_total + P3_total), 0.0)
    
    def eq(f):
        if not (0 <= f < 1):
            return float("inf")
        # Sum each individual outcome
        s = (P5 * b5) / (1 + f * b5)
        s += sum((Pi * bi) / (1 + f * bi) for Pi, bi in zip(P4_list, b4_list))
        s += sum((Pi * bi) / (1 + f * bi) for Pi, bi in zip(P3_list, b3_list))
        return s - (P_L / (1 - f))

    f_star = _solve_kelly(eq)
    
    # --- Context for UI (Aggregate view of accurate data) ---
    # Calculate probability-weighted average payouts for display
    b4_avg = sum(p * b for p, b in zip(P4_list, b4_list)) / P4_total if P4_total > 0 else 0
    b3_avg = sum(p * b for p, b in zip(P3_list, b3_list)) / P3_total if P3_total > 0 else 0

    ctx = {"P5": P5, "P4": P4_total, "P3": P3_total, "P_L": P_L, "b5": b5, "b4": b4_avg, "b3": b3_avg}
    return f_star, ctx

# ── 6-leg Kelly ──────────────────────────────────────────────────────────────
def calculate_6_leg_kelly(probabilities, nets, mults):
    p = probabilities
    q = [1 - x for x in p]

    # --- Accurate Kelly Calculation (Iterative) ---
    # 1. 6 of 6 (1 outcome)
    P6 = prod(p)
    b6 = nets[0] * prod(mults)

    # 2. 5 of 6 (6 outcomes)
    P5_list, b5_list = [], []
    for combo in combinations(range(6), 5):
        miss_idx = (set(range(6)) - set(combo)).pop()
        prob = prod(p[i] for i in combo) * q[miss_idx]
        payout_mult = prod(mults[i] for i in combo)
        
        P5_list.append(prob)
        b5_list.append(nets[1] * payout_mult)

    # 3. 4 of 6 (15 outcomes)
    P4_list, b4_list = [], []
    for combo in combinations(range(6), 4):
        miss_indices = set(range(6)) - set(combo)
        prob = prod(p[i] for i in combo) * prod(q[j] for j in miss_indices)
        payout_mult = prod(mults[i] for i in combo)
        
        P4_list.append(prob)
        b4_list.append(nets[2] * payout_mult)
    
    # 4. Total probabilities for loss and context
    P5_total = sum(P5_list)
    P4_total = sum(P4_list)
    P_L = max(1 - (P6 + P5_total + P4_total), 0.0)

    def eq(f):
        if not (0 <= f < 1):
            return float("inf")
        # Sum each individual outcome
        s = (P6 * b6) / (1 + f * b6)
        s += sum((Pi * bi) / (1 + f * bi) for Pi, bi in zip(P5_list, b5_list))
        s += sum((Pi * bi) / (1 + f * bi) for Pi, bi in zip(P4_list, b4_list))
        return s - (P_L / (1 - f))

    f_star = _solve_kelly(eq)

    # --- Context for UI (Aggregate view of accurate data) ---
    # Calculate probability-weighted average payouts for display
    b5_avg = sum(p * b for p, b in zip(P5_list, b5_list)) / P5_total if P5_total > 0 else 0
    b4_avg = sum(p * b for p, b in zip(P4_list, b4_list)) / P4_total if P4_total > 0 else 0

    ctx = {"P6": P6, "P5": P5_total, "P4": P4_total, "P_L": P_L, "b6": b6, "b5": b5_avg, "b4": b4_avg}
    return f_star, ctx

# ── Streamlit UI ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Kelly Criterion Calculator", layout="wide")
st.title("📈 Multi-leg Kelly Criterion Calculator")

legs = st.sidebar.selectbox("Number of legs", [4, 5, 6])

with st.form("kelly_form"):
    st.subheader(f"Enter odds or probabilities for {legs}-leg parlay")
    cols = st.columns(legs)
    odds_raw = [cols[i].text_input(f"Leg {i+1}", value="-110") for i in range(legs)]
    probabilities = [parse_odds_input(x) for x in odds_raw]

    st.subheader("Enter multipliers (default 1.0)")
    mcols = st.columns(legs)
    mults = [mcols[i].number_input(f"Mult Leg {i+1}", value=1.0, format="%.2f") for i in range(legs)]

    st.subheader("Enter gross payouts (includes stake)")
    if legs == 4:
        gross4 = st.number_input("4/4 gross", value=7.2, format="%.2f")
        gross3 = st.number_input("3/4 gross", value=1.8, format="%.2f")
        nets = [gross_to_net(gross4), gross_to_net(gross3)]
    elif legs == 5:
        g5 = st.number_input("5/5 gross", value=10.0, format="%.2f")
        g4 = st.number_input("4/5 gross", value=2.0, format="%.2f")
        g3 = st.number_input("3/5 gross", value=0.5, format="%.2f")
        nets = list(map(gross_to_net, [g5, g4, g3]))
    else:  # legs == 6
        g6 = st.number_input("6/6 gross", value=20.0, format="%.2f")
        g5 = st.number_input("5/6 gross", value=2.0, format="%.2f")
        g4 = st.number_input("4/6 gross", value=0.5, format="%.2f")
        nets = list(map(gross_to_net, [g6, g5, g4]))

    bankroll = st.number_input("Enter your bankroll", value=1000.0, format="%.2f")

    submitted = st.form_submit_button("Calculate")

if submitted:
    # Compute Kelly and build outcome rows
    if legs == 4:
        f_star, ctx = calculate_4_leg_kelly(probabilities, mults, nets[0], nets[1])
        rows = [
            ("4 of 4", ctx["P4"], ctx["b4"]),
            *[(f"3 of 4 (miss leg {i+1})", ctx["P3"][i], ctx["b3"][i]) for i in range(4)],
            ("Lose", ctx["P_L"], -1.0),
        ]
    elif legs == 5:
        f_star, ctx = calculate_5_leg_kelly(probabilities, nets, mults)
        rows = [
            ("5 of 5", ctx["P5"], ctx["b5"]),
            ("4 of 5", ctx["P4"], ctx["b4"]),
            ("3 of 5", ctx["P3"], ctx["b3"]),
            ("Lose (0-2 hits)", ctx["P_L"], -1.0),
        ]
    else:  # legs == 6
        f_star, ctx = calculate_6_leg_kelly(probabilities, nets, mults)
        rows = [
            ("6 of 6", ctx["P6"], ctx["b6"]),
            ("5 of 6", ctx["P5"], ctx["b5"]),
            ("4 of 6", ctx["P4"], ctx["b4"]),
            ("Lose (≤3 hits)", ctx["P_L"], -1.0),
        ]

    # Results header
    st.subheader("📊 Results")

    # Outcome table
    df = pd.DataFrame(rows, columns=["Outcome", "Probability", "Net Payout"])
    st.dataframe(
        df.style.format({"Probability": "{:.2%}", "Net Payout": "${:.2f}"}),
        use_container_width=True
    )

    # Expected value (% of stake) from rows (per $1 stake)
    ev = sum(prob * payout for _, prob, payout in rows)
    ev_pct = ev * 100

    # Kelly fractions and stakes
    quarter_kelly = f_star / 4
    full_stake = f_star * bankroll
    quarter_stake = quarter_kelly * bankroll

    # Expected log growth (only valid when 1 + f * payout > 0)
    growth_full = sum(
        prob * math.log(1 + f_star * payout)
        for _, prob, payout in rows
        if (1 + f_star * payout) > 0
    )
    growth_quarter = sum(
        prob * math.log(1 + quarter_kelly * payout)
        for _, prob, payout in rows
        if (1 + quarter_kelly * payout) > 0
    )

    # Display metrics
    if f_star > 0:
        st.success(f"Optimal Kelly fraction: {f_star:.2%} of bankroll")
    else:
        st.error("No positive edge → 0% stake")

    st.info(
        f"Expected Value: {ev_pct:.2f}% of stake\n\n"
        f"Full Kelly fraction: {f_star:.2%} → Stake ${full_stake:.2f}\n\n"
        f"Quarter Kelly fraction: {quarter_kelly:.2%} → Stake ${quarter_stake:.2f}\n\n"
        f"Expected log growth (Full Kelly): {growth_full:.6f}\n\n"
        f"Expected log growth (Quarter Kelly): {growth_quarter:.6f}"
    )
