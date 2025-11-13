import streamlit as st
import scipy.optimize as optimize
from itertools import combinations
from math import prod
import pandas as pd
import math

# ── Helper: convert gross return (stake included) → net return
def gross_to_net(gross):
    return float(gross) - 1.0

# ── Helper: implied probability from American odds
def american_to_prob(odds):
    odds = float(odds)
    if odds < 0:
        p = -odds / (-odds + 100.0)
    else:
        p = 100.0 / (odds + 100.0)
    return min(max(p, 1e-12), 1 - 1e-12)

# ── Helper: auto-detect odds or probability input
def parse_odds_input(x_raw):
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

# ── Solver: improved _solve_kelly with candidate evaluation
def _solve_kelly_and_pick(kelly_eq, expected_log_growth, positive_b, eps=1e-12,
                          safety_fraction=0.999999, f_upper_cap=0.9999):
    if positive_b:
        f_upper = min(f_upper_cap, min((1.0 - 1e-9) / b for b in positive_b))
    else:
        f_upper = f_upper_cap

    try:
        y0 = kelly_eq(0.0)
    except Exception:
        y0 = float("nan")
    try:
        y1 = kelly_eq(f_upper)
    except Exception:
        y1 = float("nan")

    if not (math.isfinite(y0) and math.isfinite(y1)):
        return 0.0, {"method": "invalid_endpoints", "f_upper": f_upper}

    candidates = {"zero": 0.0}
    f_bound = f_upper * safety_fraction
    candidates["bound"] = f_bound

    root = None
    if abs(y0) <= eps:
        root = 0.0
        candidates["root"] = root
    elif y0 > 0 and y1 > 0:
        root = None
    elif y0 <= 0 and y1 <= 0:
        root = None
    else:
        try:
            sol = optimize.root_scalar(kelly_eq, bracket=[0.0, f_upper], method="brentq")
            if sol.converged and 0 <= sol.root < 1:
                root = sol.root
                candidates["root"] = root
        except Exception:
            root = None

    evals = {}
    for name, f in candidates.items():
        try:
            val = expected_log_growth(f)
        except Exception:
            val = float("-inf")
        if not math.isfinite(val):
            val = float("-inf")
        evals[name] = (f, val)

    best_name, (f_pick, g_pick) = max(evals.items(), key=lambda kv: (math.isfinite(kv[1][1]), kv[1][1]))

    info = {
        "f_upper": f_upper,
        "candidates": {k: v for k, v in candidates.items()},
        "evals": {k: v[1] for k, v in evals.items()},
        "chosen": best_name,
        "root_found": root is not None,
        "root": root,
    }
    return f_pick, info

# ── 4-leg Kelly
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

    wins = [(P4, b4, tuple(range(4)))]
    for Pi, bi, idx in zip(P3, b3, range(4)):
        wins.append((Pi, bi, (idx,)))

    losses = []
    for k in range(0, 3):
        for combo in combinations(range(4), k):
            prob = prod(probabilities[i] for i in combo) * prod((1 - probabilities[j]) for j in set(range(4)) - set(combo))
            losses.append((prob, combo))
    P_L_enum = sum(prob for prob, _ in losses)

    def eq(f):
        if not (0 <= f < 1):
            return float("inf")
        s = 0.0
        for P, b, _ in wins:
            denom = 1 + f * b
            if denom <= 0:
                return float("inf")
            s += (P * b) / denom
        loss_denom = 1 - f
        if loss_denom <= 0:
            return float("inf")
        s -= sum(prob / loss_denom for prob, _ in losses)
        return s

    positive_b = [b for _, b, _ in wins if b > 0]

    def expected_log_growth(f):
        terms = []
        for P, b, _ in wins:
            denom = 1 + f * b
            if denom <= 0:
                return float("-inf")
            terms.append(P * math.log(denom))
        denom_loss = 1 - f
        if denom_loss <= 0:
            return float("-inf")
        terms.append(P_L_enum * math.log(denom_loss))
        return sum(terms)

    f_star, solver_info = _solve_kelly_and_pick(eq, expected_log_growth, positive_b)
    ev = sum(P * b for P, b, _ in wins) - P_L_enum

    ctx = {
        "P4": P4, "P3": P3, "P_L": P_L,
        "b4": b4, "b3": b3,
        "wins": wins, "losses": losses,
        "f_info": solver_info
    }
    return f_star, ctx

# ── 5-leg Kelly
def calculate_5_leg_kelly(probabilities, nets, mults):
    wins = []
    for combo in combinations(range(5), 5):
        prob = prod(probabilities[i] for i in combo)
        payout = nets[0] * prod(mults[i] for i in combo)
        wins.append((prob, payout, combo))
    for combo in combinations(range(5), 4):
        miss_idx = (set(range(5)) - set(combo)).pop()
        prob = prod(probabilities[i] for i in combo) * (1 - probabilities[miss_idx])
        payout = nets[1] * prod(mults[i] for i in combo)
        wins.append((prob, payout, combo))
    for combo in combinations(range(5), 3):
        miss_indices = set(range(5)) - set(combo)
        prob = prod(probabilities[i] for i in combo) * prod((1 - probabilities[j]) for j in miss_indices)
        payout = nets[2] * prod(mults[i] for i in combo)
        wins.append((prob, payout, combo))

    losses = []
    for k in range(0, 3):
        for combo in combinations(range(5), k):
            prob = prod(probabilities[i] for i in combo) * prod((1 - probabilities[j]) for j in set(range(5)) - set(combo))
            losses.append((prob, combo))
    P_L = sum(prob for prob, _ in losses)

    def eq(f):
        if not (0 <= f < 1):
            return float("inf")
        s = 0.0
        for P, b, _ in wins:
            denom = 1 + f * b
            if denom <= 0:
                return float("inf")
            s += (P * b) / denom
        loss_denom = 1 - f
        if loss_denom <= 0:
            return float("inf")
        s -= sum(prob / loss_denom for prob, _ in losses)
        return s

    positive_b = [b for _, b, _ in wins if b > 0]

    def expected_log_growth(f):
        terms = []
        for P, b, _ in wins:
            denom = 1 + f * b
            if denom <= 0:
                return float("-inf")
            terms.append(P * math.log(denom))
        denom_loss = 1 - f
        if denom_loss <= 0:
            return float("-inf")
        terms.append(P_L * math.log(denom_loss))
        return sum(terms)

    f_star, solver_info = _solve_kelly_and_pick(eq, expected_log_growth, positive_b)
    ev = sum(P * b for P, b, _ in wins) - P_L

    def tier_prob(k):
        total = 0.0
        for combo in combinations(range(5), k):
            prob = prod(probabilities[i] for i in combo) * prod((1 - probabilities[j]) for j in set(range(5)) - set(combo))
            total += prob
        return total

    P5 = tier_prob(5)
    P4 = tier_prob(4)
    P3 = tier_prob(3)
    wavg = lambda k: (
        sum(P * b for (P, b, combo) in wins if len(combo) == k) /
        max(1e-18, sum(P for (P, b, combo) in wins if len(combo) == k))
    )

    return {
        "f_star": f_star,
        "EV": ev,
        "growth_full": expected_log_growth(f_star),
        "growth_quarter": expected_log_growth(f_star / 4.0),
        "wins": wins,
        "losses": losses,
        "P_L": P_L,
        "tiers": {
            "P5": P5, "P4": P4, "P3": P3,
            "b5_avg": wavg(5), "b4_avg": wavg(4), "b3_avg": wavg(3)
        },
        "f_info": solver_info
    }

# ── 6-leg Kelly
def calculate_6_leg_kelly(probabilities, nets, mults):
    wins = []
    for combo in combinations(range(6), 6):
        prob = prod(probabilities[i] for i in combo)
        payout = nets[0] * prod(mults[i] for i in combo)
        wins.append((prob, payout, combo))
    for combo in combinations(range(6), 5):
        miss_idx = (set(range(6)) - set(combo)).pop()
        prob = prod(probabilities[i] for i in combo) * (1 - probabilities[miss_idx])
        payout = nets[1] * prod(mults[i] for i in combo)
        wins.append((prob, payout, combo))
    for combo in combinations(range(6), 4):
        miss_indices = set(range(6)) - set(combo)
        prob = prod(probabilities[i] for i in combo) * prod((1 - probabilities[j]) for j in miss_indices)
        payout = nets[2] * prod(mults[i] for i in combo)
        wins.append((prob, payout, combo))

    losses = []
    for k in range(0, 4):
        for combo in combinations(range(6), k):
            prob = prod(probabilities[i] for i in combo) * prod((1 - probabilities[j]) for j in set(range(6)) - set(combo))
            losses.append((prob, combo))
    P_L = sum(prob for prob, _ in losses)

    def eq(f):
        if not (0 <= f < 1):
            return float("inf")
        s = 0.0
        for P, b, _ in wins:
            denom = 1 + f * b
            if denom <= 0:
                return float("inf")
            s += (P * b) / denom
        loss_denom = 1 - f
        if loss_denom <= 0:
            return float("inf")
        s -= sum(prob / loss_denom for prob, _ in losses)
        return s

    positive_b = [b for _, b, _ in wins if b > 0]

    def expected_log_growth(f):
        terms = []
        for P, b, _ in wins:
            denom = 1 + f * b
            if denom <= 0:
                return float("-inf")
            terms.append(P * math.log(denom))
        denom_loss = 1 - f
        if denom_loss <= 0:
            return float("-inf")
        terms.append(P_L * math.log(denom_loss))
        return sum(terms)

    f_star, solver_info = _solve_kelly_and_pick(eq, expected_log_growth, positive_b)
    ev = sum(P * b for P, b, _ in wins) - P_L

    def tier_prob(k):
        total = 0.0
        for combo in combinations(range(6), k):
            prob = prod(probabilities[i] for i in combo) * prod((1 - probabilities[j]) for j in set(range(6)) - set(combo))
            total += prob
        return total

    P6 = tier_prob(6)
    P5 = tier_prob(5)
    P4 = tier_prob(4)
    wavg = lambda k: (
        sum(P * b for (P, b, combo) in wins if len(combo) == k) /
        max(1e-18, sum(P for (P, b, combo) in wins if len(combo) == k))
    )

    return {
        "f_star": f_star,
        "EV": ev,
        "growth_full": expected_log_growth(f_star),
        "growth_quarter": expected_log_growth(f_star / 4.0),
        "wins": wins,
        "losses": losses,
        "P_L": P_L,
        "tiers": {
            "P6": P6, "P5": P5, "P4": P4,
            "b6_avg": wavg(6), "b5_avg": wavg(5), "b4_avg": wavg(4)
        },
        "f_info": solver_info
    }

# ── Streamlit UI
st.set_page_config(page_title="Kelly Criterion Calculator", layout="wide")
st.title("📈 Multi-leg Kelly Criterion Calculator (Payout-multiplier)")

legs = st.sidebar.selectbox("Number of legs", [4, 5, 6])

with st.form("kelly_form"):
    st.subheader(f"Enter odds or probabilities for {legs}-leg parlay")
    cols = st.columns(legs)
    odds_raw = [cols[i].text_input(f"Leg {i+1}", value="-110") for i in range(legs)]
    probabilities = [parse_odds_input(x) for x in odds_raw]

    st.subheader("Enter multipliers (default 1.0)")
    mcols = st.columns(legs)
    mults = [mcols[i].number_input(f"Mult Leg {i+1}", value=1.0, format="%.6f") for i in range(legs)]

    st.subheader("Enter gross payouts (includes stake)")
    if legs == 4:
        gross4 = st.number_input("4/4 gross", value=7.2, format="%.4f")
        gross3 = st.number_input("3/4 gross", value=1.8, format="%.4f")
        nets = [gross_to_net(gross4), gross_to_net(gross3)]
    elif legs == 5:
        g5 = st.number_input("5/5 gross", value=10.0, format="%.4f")
        g4 = st.number_input("4/5 gross", value=2.0, format="%.4f")
        g3 = st.number_input("3/5 gross", value=0.5, format="%.4f")
        nets = list(map(gross_to_net, [g5, g4, g3]))
    else:  # legs == 6
        g6 = st.number_input("6/6 gross", value=20.0, format="%.4f")
        g5 = st.number_input("5/6 gross", value=2.0, format="%.4f")
        g4 = st.number_input("4/6 gross", value=0.5, format="%.4f")
        nets = list(map(gross_to_net, [g6, g5, g4]))

    # NEW: overall payout multiplier (applies to all winning payouts)
    st.subheader("Overall payout multiplier (applies to all gross payouts)")
    payout_multiplier = st.number_input("Payout multiplier", value=1.0, min_value=1.0, format="%.6f")

    # Apply payout multiplier to nets (every payout that occurs will be scaled)
    nets = [n * payout_multiplier for n in nets]

    bankroll = st.number_input("Enter your bankroll", value=1000.0, format="%.2f")

    submitted = st.form_submit_button("Calculate")

if submitted:
    boundary_warning = False
    solver_info = None

    if legs == 4:
        f_star, ctx = calculate_4_leg_kelly(probabilities, mults, nets[0], nets[1])
        solver_info = ctx.get("f_info")
        rows = [
            ("4 of 4", ctx["P4"], ctx["b4"]),
            *[(f"3 of 4 (miss leg {i+1})", ctx["P3"][i], ctx["b3"][i]) for i in range(4)],
            ("Lose", ctx["P_L"], -1.0),
        ]
        ev = sum(prob * payout for _, prob, payout in rows)
        growth_full = sum(
            prob * math.log(1 + f_star * payout)
            for _, prob, payout in rows
            if (1 + f_star * payout) > 0
        )
        growth_quarter = sum(
            prob * math.log(1 + (f_star / 4.0) * payout)
            for _, prob, payout in rows
            if (1 + (f_star / 4.0) * payout) > 0
        )
    elif legs == 5:
        result = calculate_5_leg_kelly(probabilities, nets, mults)
        f_star = result["f_star"]
        solver_info = result.get("f_info")
        tiers = result["tiers"]
        rows = [
            ("5 of 5", tiers["P5"], tiers["b5_avg"]),
            ("4 of 5", tiers["P4"], tiers["b4_avg"]),
            ("3 of 5", tiers["P3"], tiers["b3_avg"]),
            ("Lose (0–2 hits)", result["P_L"], -1.0),
        ]
        ev = result["EV"]
        growth_full = result["growth_full"]
        growth_quarter = result["growth_quarter"]
    else:  # legs == 6
        result = calculate_6_leg_kelly(probabilities, nets, mults)
        f_star = result["f_star"]
        solver_info = result.get("f_info")
        tiers = result["tiers"]
        rows = [
            ("6 of 6", tiers["P6"], tiers["b6_avg"]),
            ("5 of 6", tiers["P5"], tiers["b5_avg"]),
            ("4 of 6", tiers["P4"], tiers["b4_avg"]),
            ("Lose (≤3 hits)", result["P_L"], -1.0),
        ]
        ev = result["EV"]
        growth_full = result["growth_full"]
        growth_quarter = result["growth_quarter"]

    if solver_info is not None:
        chosen = solver_info.get("chosen")
        if chosen == "bound":
            boundary_warning = True

    st.subheader("📊 Results")

    df = pd.DataFrame(rows, columns=["Outcome", "Probability", "Net Payout"])
    st.dataframe(
        df.style.format({"Probability": "{:.6%}", "Net Payout": "${:.6f}"}),
        use_container_width=True
    )

    ev_pct = ev * 100.0

    quarter_kelly = f_star / 4.0
    full_stake = f_star * bankroll
    quarter_stake = quarter_kelly * bankroll

    growth_full_bps = (math.exp(growth_full) - 1) * 10000.0
    growth_quarter_bps = (math.exp(growth_quarter) - 1) * 10000.0

    if f_star > 0:
        st.success(f"Optimal Kelly fraction: {f_star:.6%} of bankroll")
    else:
        st.error("No positive edge → 0% stake")

    if boundary_warning:
        st.warning(
            "The solver chose a boundary-optimal stake (the Kelly FOC increases up to the safe "
            "boundary). This means expected log-growth rises as you approach the singularity; "
            "recommended f is set just inside that boundary. Small changes in probabilities or "
            "payouts could materially change this recommendation."
        )

    st.info(
        f"Payout multiplier applied: {payout_multiplier:.6f}x\n\n"
        f"Expected Value: {ev_pct:.6f}% of stake\n\n"
        f"Full Kelly fraction: {f_star:.6%} → Stake ${full_stake:.6f}\n"
        f"Expected bankroll growth (Full Kelly): {growth_full_bps:.6f} BPS\n"
        f" (Log growth: {growth_full:.12f})\n\n"
        f"Quarter Kelly fraction: {quarter_kelly:.6%} → Stake ${quarter_stake:.6f}\n"
        f"Expected bankroll growth (Quarter Kelly): {growth_quarter_bps:.6f} BPS\n"
        f" (Log growth: {growth_quarter:.12f})"
    )
