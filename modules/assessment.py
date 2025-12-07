#    ___    _   ___  _    _    
#   | _ \  /_\ |   \(_)__| |_  
#   |   / / _ \| |) | (_-< ' \ 
#   |_|_\/_/ \_\___/|_/__/_||_| V1
#                               
#   R A D I S H | RISK ANALYSIS DASHBOARD
#   CS4001 DATA VISUALIZATION DESIGN - PROJECT 
#   TEAM OPPORTUNISTS | Shreya Farhat Puneet
#   SEP 2025 
# ----------------------------------------------
#   Support: radish@ds.study.iitm.ac.in
# ----------------------------------------------

import streamlit as st
import plotly.express as px
import pandas as pd
from modules import data_engine

def show(df):
    st.header("Credit Assessment & Decisioning")
    
    # Get Currency Symbol from Preferences
    currency = data_engine.get_currency_symbol()
    
    # ---------------------------------------------------------
    # PB 20251206 TABS
    # ---------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["Snapshot & Cohorts", "Bias Auditor", "Applicant Details"])
    
    # =========================================================
    # TAB 1: COHORTS
    # =========================================================
    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Cohort Analysis")
            group_col = st.selectbox("Group By", ['NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE', 'CODE_GENDER'])
            cohort_risk = df.groupby(group_col)['TARGET'].mean().reset_index()
            fig = px.bar(cohort_risk, x=group_col, y='TARGET', color='TARGET', title="Default Risk by Cohort")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Storytelling Mode")
            # Narrative generation logic
            high_risk_segment = cohort_risk.sort_values('TARGET', ascending=False).iloc[0]
            st.markdown(f"""
            ### 🚨 Risk Alert
            Analysis indicates that the **{high_risk_segment[group_col]}** segment is currently the highest risk cohort with a default rate of **{high_risk_segment['TARGET']:.2%}**. 
            
            **Recommendation:** Enhanced due diligence is required for applicants in this category. Check *Previous Refusals* in the 'Applicant Details' tab.
            """)

    # =========================================================
    # TAB 2: BIAS
    # =========================================================
    with tab2:
        st.subheader("Bias Auditor")
        st.write("Checking for demographic disparities in Approval/Default rates.")
        fig = px.histogram(df, x='CODE_GENDER', color='TARGET', barmode='group', 
                           histnorm='percent', title="Gender Disparity in Risk Outcomes")
        st.plotly_chart(fig, use_container_width=True)

    # =========================================================
    # TAB 3: DETAILS
    # =========================================================
    with tab3:
        st.subheader("Applicant Deep Dive")
        # Default to first ID in the list if available
        default_id = int(df['SK_ID_CURR'].iloc[0]) if not df.empty else 0
        app_id = st.number_input("Enter Applicant ID (SK_ID_CURR)", value=default_id)
        
        # Get Record
        record = df[df['SK_ID_CURR'] == app_id]
        
        if not record.empty:
            st.success(f"Applicant Found: {app_id}")
            
            # Key Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Income", f"{currency}{record['AMT_INCOME_TOTAL'].values[0]:,.0f}")
            c2.metric("Credit Amount", f"{currency}{record['AMT_CREDIT'].values[0]:,.0f}")
            
            prev_count = int(record['PREV_APP_COUNT'].values[0]) if 'PREV_APP_COUNT' in record.columns else 0
            c3.metric("Prev. Applications", prev_count)
            
            c4.metric("Risk Label (Target)", int(record['TARGET'].values[0]))
            
            st.write("**Current Application Data:**")
            st.dataframe(record)
            
            st.write("**Previous Application History:**")
            history = data_engine.get_applicant_history(app_id)
            if not history.empty:
                # Format currency columns in history if they exist
                format_dict = {}
                for col in ['AMT_APPLICATION', 'AMT_CREDIT', 'AMT_ANNUITY']:
                    if col in history.columns:
                        format_dict[col] = f"{currency}{{:.0f}}"
                
                st.dataframe(history.style.format(format_dict))
            else:
                st.warning("No historical data found for this ID.")
        else:
            st.error("Applicant ID not found.")