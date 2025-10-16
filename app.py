import streamlit as st
import scipy.optimize as optimize
from itertools import combinations
from math import prod
import pandas as pd

# ── Helper: convert gross return (stake included) → net return ────────────────
def gross_to_net(gross):
    return float(gross) - 1.0

# ── Helper: implied probability from American odds ───────────────────────────
def american_to_prob(odds):
    if odds is None:
        return 0.5
    odds = float(odds)
    if odds < 0:
        p = -odds / (-odds + 100.0)
    elif odds > 0:
        p = 100.0 / (odds + 100.0)
    else:
        p = 0.5
    return min(max(p, 1e-12), 1-1e-12)

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
def calculate_4_leg_kelly(odds, mults, net4, net3):
    p = [american_to_prob(o) for o in odds]
    q = [1-x for x in p]

    P4 = prod(p)
    P3 = [
        p[0]*p[1]*p[2]*q[3],
        p[0]*p[1]*q[2]*p[3],
        p[0]*q[1]*p[2]*p[3],
        q[0]*p[1]*p[2]*p[3],
    ]
    P_L = max(1 - (P4 + sum(P3)), 0.0)

    b4 = net4 * prod(mults)
    b3 = [
        net3 * mults[0]*mults[1]*mults[2],
        net3 * mults[0]*mults[1]*mults[3],
        net3 * mults[0]*mults[2]*mults[3],
        net3 * mults[1]*mults[2]*mults[3],
    ]

    def eq(f):
        if not (0 <= f < 1):
            return float("inf")
        win_sum = (P4 * b4) / (1 + f*b4)
        for Pi, bi in zip(P3, b3):
            win_sum += (Pi * bi) / (1 + f*bi)
        return win_sum - (P_L / (1 - f))

    f_star = _solve_kelly(eq)
    ctx = {"P4": P4, "P3": P3, "P_L": P_L, "b4": b4, "b3": b3}
    return f_star, ctx

# ── 5-leg Kelly ──────────────────────────────────────────────────────────────
def calculate_5_leg_kelly(odds, nets):
    p = [american_to_prob(o) for o in odds]
    q = [1-x for x in p]
    def P(k):
        return sum(
            prod(p[i] for i in combo) * prod(q[j] for j in set(range(5)) - set(combo))
            for combo in combinations(range(5), k)
        )
    P5, P4, P3 = P(5), P(4), P(3)
    P_L = max(1 - (P5 + P4 + P3), 0.0)

    def eq(f):
        if not (0 <= f < 1):
            return float("inf")
        s = (P5 * nets[0])/(1+f*nets[0]) + (P4 * nets[1])/(1+f*nets[1]) + (P3 * nets[2])/(1+f*nets[2])
        return s - (P_L / (1 - f))

    f_star = _solve_kelly(eq)
    ctx = {"P5": P5, "P4": P4, "P3": P3, "P_L": P_L}
    return f_star, ctx

# ── 6-leg Kelly ──────────────────────────────────────────────────────────────
def calculate_6_leg_kelly(odds, nets):
    p = [american_to_prob(o) for o in odds]
    q = [1-x for x in p]
    def P(k):
        return sum(
            prod(p[i] for i in combo) * prod(q[j] for j in set(range(6)) - set(combo))
            for combo in combinations(range(6), k)
        )
    P6, P5, P4 = P(6), P(5), P(4)
    P_L = max(1 - (P6 + P5 + P4), 0.0)

    def eq(f):
        if not (0 <= f < 1):
            return float("inf")
        s = (P6 * nets[0])/(1+f*nets[0]) + (P5 * nets[1])/(1+f*nets[1]) + (P4 * nets[2])/(1+f*nets[2])
        return s - (P_L / (1 - f))

    f_star = _solve_kelly(eq)
    ctx = {"P6": P6, "P5": P5, "P4": P4, "P_L": P_L}
    return f_star, ctx

# ── Streamlit UI ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Kelly Criterion Calculator", layout="wide")
st.title("📈 Multi-leg Kelly Criterion Calculator")

legs = st.sidebar.selectbox("Number of legs", [4, 5, 6])

with st.form("kelly_form"):
    st.subheader(f"Enter odds for {legs}-leg parlay")
    cols = st.columns(legs)
    odds = [cols[i].number_input(f"Odds Leg {i+1}", value=-110) for i in range(legs)]

    if legs == 4:
        st.subheader("Enter multipliers (default 1.0)")
        mcols = st.columns(4)
        mults = [mcols[i].number_input(f"Mult Leg {i+1}", value=1.0, format="%.2f") for i in range(4)]
    else:
        mults = [1.0]*legs

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
    else:  # legs==6
        g6 = st.number_input("6/6 gross", value=20.0, format="%.2f")
        g5 = st.number_input("5/6 gross", value=2.0, format="%.2f")
        g4 = st.number_input("4/6 gross", value=0.5, format="%.2f")
        nets = list(map(gross_to_net, [g6, g5, g4]))

    submitted = st.form_submit_button("Calculate Kelly stake")

if submitted:
    if legs == 4:
        f_star, ctx = calculate_4_leg_kelly(odds, mults, nets[0], nets[1])
        rows = [
            ("4 of 4", ctx["P4"], ctx["b4"]),
            *[(f"3 of 4 (miss leg {i+1})", ctx["P3"][i], ctx["b3"][i]) for i in range(4)],
            ("Lose", ctx["P_L"], -1.0)
        ]
    elif legs == 5:
        f_star, ctx = calculate_5_leg_kelly(odds, nets)
        rows = [
            ("5 of 5", ctx["P5"], nets[0]),
            ("4 of
