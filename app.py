import streamlit as st
import scipy.optimize as optimize
from itertools import combinations
from math import prod
import pandas as pd

# ==============================================================================
# --- 1. HELPER & CALCULATION FUNCTIONS (FROM YOUR SCRIPTS) ---
# We've copied the core logic from your files into this one app.
# ==============================================================================

def american_to_prob(odds):
    """Converts American odds to implied probability."""
    if odds is None: return None
    if odds < 0: return -odds / (-odds + 100)
    elif odds > 0: return 100 / (odds + 100)
    return 0.5

# --- Logic from generalized_kelly.py ---
def calculate_4_leg_kelly(odds1, odds2, odds3, odds4, m1, m2, m3, m4, base_payout_4_4, base_payout_3_4):
    p1, p2, p3, p4 = american_to_prob(odds1), american_to_prob(odds2), american_to_prob(odds3), american_to_prob(odds4)
    q1, q2, q3, q4 = 1 - p1, 1 - p2, 1 - p3, 1 - p4
    P_4_4 = p1 * p2 * p3 * p4
    P_3A, P_3B, P_3C, P_3D = p1*p2*p3*q4, p1*p2*q3*p4, p1*q2*p3*p4, q1*p2*p3*p4
    P_L = 1 - (P_4_4 + P_3A + P_3B + P_3C + P_3D)
    b_4_4 = base_payout_4_4 * m1 * m2 * m3 * m4
    b_3A, b_3B, b_3C, b_3D = base_payout_3_4*m1*m2*m3, base_payout_3_4*m1*m2*m4, base_payout_3_4*m1*m3*m4, base_payout_3_4*m2*m3*m4
    
    def kelly_eq(f):
        if abs(1.0 - f) < 1e-9: return float('inf')
        win_outcomes = [(P_4_4, b_4_4), (P_3A, b_3A), (P_3B, b_3B), (P_3C, b_3C), (P_3D, b_3D)]
        win_sum = sum((p * b) / (1 + f * b) for p, b in win_outcomes)
        return win_sum - (P_L / (1 - f))

    context = locals()
    if kelly_eq(0) <= 0: return 0, context
    try:
        sol = optimize.root_scalar(kelly_eq, bracket=[0.0, 0.9999])
        return sol.root, context
    except ValueError: return 0, context

# --- Logic from flex5kelly.py ---
def calculate_5_leg_kelly(odds_in, payouts):
    probs = [american_to_prob(o) for o in odds_in]
    qrobs = [1 - p for p in probs]
    P_5_5 = prod(probs)
    P_4_5 = sum(prod(probs[j] for j in i) * qrobs[list(set(range(5)) - set(i))[0]] for i in combinations(range(5), 4))
    P_3_5 = sum(prod(probs[j] for j in i) * prod(qrobs[k] for k in set(range(5)) - set(i)) for i in combinations(range(5), 3))
    P_L = 1 - (P_5_5 + P_4_5 + P_3_5)

    def kelly_eq(f):
        if abs(1.0 - f) < 1e-9: return float('inf')
        outcomes = [(P_5_5, payouts['p5']), (P_4_5, payouts['p4']), (P_3_5, payouts['p3'])]
        win_sum = sum((p * b) / (1 + f * b) for p, b in outcomes)
        return win_sum - (P_L / (1 - f))

    context = locals()
    if kelly_eq(0) <= 0: return 0, context
    try:
        sol = optimize.root_scalar(kelly_eq, bracket=[0.0, 0.9999])
        return sol.root, context
    except ValueError: return 0, context

# --- Logic from flex6kelly.py ---
def calculate_6_leg_kelly(odds_in, payouts):
    probs = [american_to_prob(o) for o in odds_in]
    qrobs = [1 - p for p in probs]
    def prob_k_of_n(k, n, p, q):
        return sum(prod(p[i] for i in wins) * prod(q[i] for i in set(range(n)) - set(wins)) for wins in combinations(range(n), k))
    
    P_6_6 = prob_k_of_n(6, 6, probs, qrobs)
    P_5_6 = prob_k_of_n(5, 6, probs, qrobs)
    P_4_6 = prob_k_of_n(4, 6, probs, qrobs)
    P_L = 1 - (P_6_6 + P_5_6 + P_4_6)

    def kelly_eq(f):
        if abs(1.0 - f) < 1e-9: return float('inf')
        outcomes = [(P_6_6, payouts['p6']), (P_5_6, payouts['p5']), (P_4_6, payouts['p4'])]
        win_sum = sum((p * b) / (1 + f * b) for p, b in outcomes)
        return win_sum - (P_L / (1 - f))

    context = locals()
    if kelly_eq(0) <= 0: return 0, context
    try:
        sol = optimize.root_scalar(kelly_eq, bracket=[0.0, 0.9999])
        return sol.root, context
    except ValueError: return 0, context


# ==============================================================================
# --- 2. STREAMLIT USER INTERFACE ---
# This part of the code creates the interactive web page.
# ==============================================================================

st.set_page_config(page_title="Kelly Criterion Calculator", layout="wide")
st.title("📈 Generalized Kelly Criterion Calculator")
st.write("Select the bet type from the sidebar to enter details and calculate the optimal stake.")

# --- Sidebar for Bet Selection ---
st.sidebar.header("Bet Configuration")
bet_type = st.sidebar.selectbox(
    "Select the number of legs for the flex parlay:",
    ["4-Leg Flex/RR", "5-Leg Flex", "6-Leg Flex"]
)

# --- Main Page Content based on Selection ---

if bet_type == "4-Leg Flex/RR":
    st.header("4-Leg Flex/Round Robin Inputs")
    with st.form("4_leg_form"):
        st.subheader("Enter American Odds")
        col1, col2, col3, col4 = st.columns(4)
        with col1: odds1 = st.number_input("Odds Leg 1", value=-110)
        with col2: odds2 = st.number_input("Odds Leg 2", value=-110)
        with col3: odds3 = st.number_input("Odds Leg 3", value=-110)
        with col4: odds4 = st.number_input("Odds Leg 4", value=-110)
        
        st.subheader("Enter Multipliers (default is 1.0)")
        col1, col2, col3, col4 = st.columns(4)
        with col1: m1 = st.number_input("Multiplier Leg 1", value=1.0, format="%.2f")
        with col2: m2 = st.number_input("Multiplier Leg 2", value=1.0, format="%.2f")
        with col3: m3 = st.number_input("Multiplier Leg 3", value=1.0, format="%.2f")
        with col4: m4 = st.number_input("Multiplier Leg 4", value=1.0, format="%.2f")

        st.subheader("Enter Base Net Payouts")
        col1, col2 = st.columns(2)
        with col1: base_payout_4_4 = st.number_input("Payout for 4/4 Correct", value=6.0, format="%.2f")
        with col2: base_payout_3_4 = st.number_input("Payout for 3/4 Correct", value=0.8, format="%.2f")
        
        submitted = st.form_submit_button("Calculate Kelly Stake")

    if submitted:
        stake, ctx = calculate_4_leg_kelly(odds1, odds2, odds3, odds4, m1, m2, m3, m4, base_payout_4_4, base_payout_3_4)
        st.subheader("📊 Results")
        if stake > 0:
            st.success(f"Optimal Stake: {stake:.2%} of your bankroll.")
        else:
            st.error("No profitable edge found. Recommended stake is 0%.")

        df_data = {
            "Outcome": ["4 of 4 Wins", "3/4 (Leg 4 Fails)", "3/4 (Leg 3 Fails)", "3/4 (Leg 2 Fails)", "3/4 (Leg 1 Fails)", "Bet Loses"],
            "Probability": [ctx['P_4_4'], ctx['P_3A'], ctx['P_3B'], ctx['P_3C'], ctx['P_3D'], ctx['P_L']],
            "Final Payout (Net)": [ctx['b_4_4'], ctx['b_3A'], ctx['b_3B'], ctx['b_3C'], ctx['b_3D'], -1]
        }
        df = pd.DataFrame(df_data)
        st.table(df.style.format({"Probability": "{:.2%}", "Final Payout (Net)": "${:.2f}"}))


elif bet_type == "5-Leg Flex":
    st.header("5-Leg Flex Parlay Inputs")
    with st.form("5_leg_form"):
        st.subheader("Enter American Odds")
        c = st.columns(5)
        odds = [c[i].number_input(f"Odds Leg {i+1}", value=-110) for i in range(5)]
        
        st.subheader("Enter Net Payouts")
        c = st.columns(3)
        p5 = c[0].number_input("Payout for 5/5 Correct", value=10.0, format="%.2f")
        p4 = c[1].number_input("Payout for 4/5 Correct", value=2.0, format="%.2f")
        p3 = c[2].number_input("Payout for 3/5 Correct", value=-0.5, format="%.2f")
        
        submitted = st.form_submit_button("Calculate Kelly Stake")

    if submitted:
        payouts = {'p5': p5, 'p4': p4, 'p3': p3}
        stake, ctx = calculate_5_leg_kelly(odds, payouts)
        st.subheader("📊 Results")
        if stake > 0:
            st.success(f"Optimal Stake: {stake:.2%} of your bankroll.")
        else:
            st.error("No profitable edge found. Recommended stake is 0%.")

        df_data = {
            "Outcome": ["5 of 5 Correct", "4 of 5 Correct", "3 of 5 Correct", "Bet Loses (0-2 hit)"],
            "Probability": [ctx['P_5_5'], ctx['P_4_5'], ctx['P_3_5'], ctx['P_L']],
            "Net Payout": [p5, p4, p3, -1]
        }
        df = pd.DataFrame(df_data)
        st.table(df.style.format({"Probability": "{:.2%}", "Net Payout": "${:.2f}"}))


elif bet_type == "6-Leg Flex":
    st.header("6-Leg Flex Parlay Inputs")
    with st.form("6_leg_form"):
        st.subheader("Enter American Odds")
        c = st.columns(6)
        odds = [c[i].number_input(f"Odds Leg {i+1}", value=-110) for i in range(6)]
        
        st.subheader("Enter Net Payouts")
        c = st.columns(3)
        p6 = c[0].number_input("Payout for 6/6 Correct", value=20.0, format="%.2f")
        p5 = c[1].number_input("Payout for 5/6 Correct", value=2.0, format="%.2f")
        p4 = c[2].number_input("Payout for 4/6 Correct", value=0.5, format="%.2f")
        
        submitted = st.form_submit_button("Calculate Kelly Stake")

    if submitted:
        payouts = {'p6': p6, 'p5': p5, 'p4': p4}
        stake, ctx = calculate_6_leg_kelly(odds, payouts)
        st.subheader("📊 Results")
        if stake > 0:
            st.success(f"Optimal Stake: {stake:.2%} of your bankroll.")
        else:
            st.error("No profitable edge found. Recommended stake is 0%.")

        df_data = {
            "Outcome": ["6 of 6 Correct", "5 of 6 Correct", "4 of 6 Correct", "Bet Loses (0-3 hit)"],
            "Probability": [ctx['P_6_6'], ctx['P_5_6'], ctx['P_4_6'], ctx['P_L']],
            "Net Payout": [p6, p5, p4, -1]
        }
        df = pd.DataFrame(df_data)
        st.table(df.style.format({"Probability": "{:.2%}", "Net Payout": "${:.2f}"}))