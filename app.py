import streamlit as st
import scipy.optimize as optimize
from itertools import combinations
from math import prod
import pandas as pd

def american_to_prob(odds):
    if odds is None: return None
    if odds < 0: return -odds / (-odds + 100)
    elif odds > 0: return 100 / (odds + 100)
    return 0.5

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

st.set_page_config(page_title="Kelly Criterion Calculator", layout="wide")
st.title("📈 Generalized Kelly Criterion Calculator")
st.write("Select the bet type from the sidebar to enter details and calculate the optimal stake.")

st.sidebar.header("Bet Configuration")
bet_type = st.sidebar.selectbox("Select the number of legs for the flex parlay:", ["4-Leg Flex/RR"])

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

        st.subheader("Enter Gross Payouts (includes stake)")
        col1, col2 = st.columns(2)
        with col1: gross_4_4 = st.number_input("Payout for 4/4 Correct (gross)", value=7.2, format="%.2f")
        with col2: gross_3_4 = st.number_input("Payout for 3/4 Correct (gross)", value=1.8, format="%.2f")

        base_payout_4_4 = gross_4_4 - 1.0
        base_payout_3_4 = gross_3_4 - 1.0

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
