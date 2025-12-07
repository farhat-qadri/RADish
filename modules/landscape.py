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
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import os
from streamlit_option_menu import option_menu
from modules import data_engine

# --- POPUP DIALOG ---
@st.dialog("Applicant Profile")
def show_applicant_popup(details, currency):
    st.markdown(f"### {details['NAME_CONTRACT_TYPE']}")
    
    status = details['NAME_CONTRACT_STATUS']
    if status == 'Refused': status_color, status_label = "red", "Declined"
    elif status == 'Approved': status_color, status_label = "green", "Approved"
    else: status_color, status_label = "orange", status
    
    st.markdown(f"**Status:** :{status_color}[{status_label}]")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Loan Amount", f"{currency}{details['AMT_APPLICATION']:,.0f}")
        st.metric("Product", details['NAME_PRODUCT_TYPE'])
    with col2:
        st.metric("Decision Day", f"{details['DAYS_DECISION']} days ago")
        st.metric("Client Type", details['NAME_CLIENT_TYPE'])
        
    st.markdown("---")
    
    if st.button("🔍 Open Full Assessment", type="primary", use_container_width=True):
        st.session_state['nav_target'] = "Risk Assessment"
        st.session_state['target_app_id'] = int(details['SK_ID_CURR'])
        st.rerun()

def show(df):
    # ---------------------------------------------------------
    # PB 20251206 CONFIG
    # ---------------------------------------------------------
    currency = data_engine.get_currency_symbol()
    
    # Initialize variables to prevent UnboundLocalError
    total_apps = 0
    total_apps_full = 0
    pct_shown = 0
    stats_df = pd.DataFrame() 

    # ---------------------------------------------------------
    # PB 20251205 2. STATE & MAIN FILTER
    # ---------------------------------------------------------
    if 'risk_selected_cats' not in st.session_state:
        st.session_state['risk_selected_cats'] = ['CODE_GENDER']
    
    if 'risk_sub_factors' not in st.session_state:
        st.session_state['risk_sub_factors'] = {}

    if 'last_chart_events' not in st.session_state:
        st.session_state['last_chart_events'] = {}

    # State for Pipeline Insights Stepper
    if 'pipeline_insight_idx' not in st.session_state:
        st.session_state['pipeline_insight_idx'] = 0
    if 'pipeline_show_insight' not in st.session_state:
        st.session_state['pipeline_show_insight'] = True

    # --- DATA PREP ---
    if 'CODE_GENDER' in df.columns:
        df['CODE_GENDER'] = df['CODE_GENDER'].replace({'M': 'Male', 'F': 'Female', 'XNA': 'Unknown'})

    # --- TOP SUB-NAVIGATION ---
    selected_sub_tab = option_menu(
        menu_title=None,
        options=["Risk Surface", "Historical Pipeline & Trends", "Feature Influence"],
        icons=["activity", "funnel", "diagram-3"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "nav-link": {"font-size": "14px", "text-align": "center", "margin":"0px 5px", "--hover-color": "#262730"},
            "nav-link-selected": {"background-color": "#17a2b8"},
        }
    )
    st.markdown("---")

    # ---------------------------------------------------------
    # PB 20251204 3. DYNAMIC SIDEBAR CONTROLS
    # ---------------------------------------------------------
    
    st.sidebar.markdown('<div class="sidebar-section">Filters</div>', unsafe_allow_html=True)
    contract_filter = st.sidebar.multiselect(
        "Contract Type", 
        df['NAME_CONTRACT_TYPE'].unique(), 
        default=df['NAME_CONTRACT_TYPE'].unique(),
        label_visibility="collapsed"
    )
    
    if not contract_filter:
        st.warning("⚠️ Please select at least one Contract Type.")
        return 
    dff = df[df['NAME_CONTRACT_TYPE'].isin(contract_filter)]

    st.sidebar.markdown('<div class="sidebar-section">Display Options</div>', unsafe_allow_html=True)
    
    if selected_sub_tab == "Risk Surface":
        use_intersection = st.sidebar.toggle("Strict Intersection", value=True, help="Match ALL risk factors vs ANY")
        show_gauges = st.sidebar.toggle("Risk Gauges", value=True)
        show_narrative = st.sidebar.toggle("Narratives", value=False)
        show_bars = st.sidebar.toggle("Detail Bars", value=True)
        
        if st.sidebar.button("🔄 Reset All", use_container_width=True):
            st.session_state['risk_sub_factors'] = {}
            st.session_state['last_chart_events'] = {}
            st.rerun()
        
    elif selected_sub_tab == "Historical Pipeline & Trends":
        max_points = st.sidebar.slider("Sample Size", 5000, 50000, 15000, step=5000)
        show_dates = st.sidebar.toggle("Show Calendar Dates", value=True, help="Toggle between Estimated Dates and Relative Days")
        
    elif selected_sub_tab == "Feature Influence":
        matrix_metric = st.sidebar.radio("Matrix Metric", ["Default Rate", "Loan Amount"], horizontal=True)
        show_trend_line = st.sidebar.toggle("Show Trend Line", value=True)

    # ---------------------------------------------------------
    # PB 20251206 4. VIEW LOGIC
    # ---------------------------------------------------------
    
    # =========================================================
    # VIEW: RISK SURFACE
    # =========================================================
    if selected_sub_tab == "Risk Surface":
        st.subheader("Explore Risk Surface and Exposure by Categories")
        st.caption("Default: **Highest Risk Factor** selected. Click bars to toggle others. Click **↺** to reset category.")
        
        # FIX: Define Global Mean HERE for access in all sub-blocks
        global_mean = dff['TARGET'].mean()

        all_cats = ['NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'CODE_GENDER', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE', 'NAME_TYPE_SUITE']
        current_cats = st.session_state['risk_selected_cats']
        available_cats = [c for c in all_cats if c not in current_cats]
        
        # --- A. STATE UPDATE & INIT ---
        for category in current_cats:
            group_stats = dff.groupby(category)['TARGET'].agg(['mean', 'count', 'sum']).reset_index()
            # FIX: Ensure 4 columns match 4 values
            group_stats.columns = [category, 'Default_Rate', 'Count', 'Defaults']
            group_stats = group_stats.sort_values('Default_Rate', ascending=True)
            
            # 1. Initialize State if Missing (Default = MAX RISK FACTOR)
            if category not in st.session_state['risk_sub_factors']:
                highest_risk_factor = group_stats.iloc[-1][category]
                st.session_state['risk_sub_factors'][category] = [highest_risk_factor]

            # 2. Check for Interaction
            chart_key = f"bar_{category}"
            current_event = st.session_state.get(chart_key, {})
            last_event = st.session_state['last_chart_events'].get(chart_key, {})
            
            if current_event != last_event:
                st.session_state['last_chart_events'][chart_key] = current_event
                
                if current_event and "selection" in current_event and current_event["selection"]["points"]:
                    # User Clicked -> Single Select
                    clicked_idx = current_event["selection"]["points"][0]["point_index"]
                    try:
                        clicked_label = group_stats.iloc[clicked_idx][category]
                        st.session_state['risk_sub_factors'][category] = [clicked_label]
                    except: pass
                else:
                    # Double Click / Deselect -> Reset to Max Risk
                    highest_risk_factor = group_stats.iloc[-1][category]
                    st.session_state['risk_sub_factors'][category] = [highest_risk_factor]

        # --- B. SUMMARY LOGIC ---
        if use_intersection:
            high_risk_mask = pd.Series([True] * len(dff), index=dff.index)
            logic_label = "Intersection"
        else:
            high_risk_mask = pd.Series([False] * len(dff), index=dff.index)
            logic_label = "Union"

        factors_found = False
        for category in current_cats:
            active = st.session_state['risk_sub_factors'].get(category, [])
            valid = [f for f in active if f in dff[category].unique()]
            
            if valid:
                if use_intersection: high_risk_mask &= dff[category].isin(valid)
                else: high_risk_mask |= dff[category].isin(valid)
                factors_found = True
            else:
                if use_intersection: high_risk_mask &= False
        
        risk_population = dff[high_risk_mask] if factors_found else pd.DataFrame()
        
        # --- C. RENDER SUMMARY ---
        if not risk_population.empty:
            summary = risk_population.groupby(['NAME_CONTRACT_TYPE', 'TARGET']).agg(Count=('TARGET', 'count'), Value=('AMT_CREDIT', 'sum')).reset_index()
            global_credit = df['AMT_CREDIT'].sum()
            global_loss = df[df['TARGET']==1]['AMT_CREDIT'].sum()
            global_loss_rate = global_loss / global_credit if global_credit > 0 else 0
            
            total_val = summary['Value'].sum()
            at_risk_val = summary[summary['TARGET']==1]['Value'].sum()
            segment_loss_rate = at_risk_val / total_val if total_val > 0 else 0
            
            def get_data(ctype, target):
                row = summary[(summary['NAME_CONTRACT_TYPE'] == ctype) & (summary['TARGET'] == target)]
                if row.empty: return 0, 0
                return row.iloc[0]['Count'], row.iloc[0]['Value']

            c0_cnt, c0_val = get_data('Cash loans', 0)
            c1_cnt, c1_val = get_data('Cash loans', 1)
            r0_cnt, r0_val = get_data('Revolving loans', 0)
            r1_cnt, r1_val = get_data('Revolving loans', 1)
            c0_el, c1_el = c0_val * segment_loss_rate, c1_val * segment_loss_rate
            r0_el, r1_el = r0_val * segment_loss_rate, r1_val * segment_loss_rate
            
            needle_deg = max(-90, min(90, (segment_loss_rate * 100 * 1.8) - 90))
            marker_deg = max(-90, min(90, (global_loss_rate * 100 * 1.8) - 90))

            html = []
            html.append('<div class="summary-card">')
            html.append(f'<h4 style="margin-top:0; color:#17a2b8;">High-Risk Segment Analysis ({logic_label})</h4>')
            html.append(f'<p style="font-size: 0.9em; color: #ccc; margin-top: -10px; margin-bottom: 20px;">This segment carries an <b>Expected Loss (EL)</b> of <b>{currency}{at_risk_val:,.0f}</b> based on a historical loss rate of <b>{segment_loss_rate:.1%}</b>.</p>')
            html.append('<div class="flex-row">')
            html.append(f'<div class="flex-col-gauge"><div style="font-size:0.9em; font-weight:bold; color:#ccc; margin-bottom:10px;">Risk Impact (Expected Loss)</div><div class="gauge-container"><div class="gauge-body"></div><div class="gauge-mask"></div><div class="gauge-needle" style="transform: rotate({needle_deg}deg);"></div><div class="gauge-marker" style="transform: rotate({marker_deg}deg);"></div><div class="gauge-value">{segment_loss_rate:.1%}</div></div><div style="font-size:0.75em; color:#888; margin-top:5px;"><span style="display:inline-block; width:8px; height:8px; background:#fff; opacity:0.6; margin-right:4px;"></span>Portfolio Avg: {global_loss_rate:.1%}</div></div>')
            html.append('<div class="flex-col-table"><table class="sleek-table"><thead><tr><th style="text-align:left;">Loan Type</th><th colspan="3" style="border-bottom: 2px solid #2ca02c;">Repayers (Target=0)</th><th colspan="3" style="border-bottom: 2px solid #d62728;">Defaulters (Target=1)</th></tr><tr><th></th><th>Count</th><th>Value</th><th title="Expected Loss">Risk Val</th><th>Count</th><th>Value</th><th title="Expected Loss">Risk Val</th></tr></thead><tbody>')
            html.append(f'<tr><td class="risk-label">Cash Loans</td><td class="risk-val val-neutral">{c0_cnt:,.0f}</td><td class="risk-val val-neutral">{currency}{c0_val:,.0f}</td><td class="risk-val val-calc">{currency}{c0_el:,.0f}</td><td class="risk-val val-risk">{c1_cnt:,.0f}</td><td class="risk-val val-risk">{currency}{c1_val:,.0f}</td><td class="risk-val val-calc">{currency}{c1_el:,.0f}</td></tr>')
            html.append(f'<tr><td class="risk-label">Revolving</td><td class="risk-val val-neutral">{r0_cnt:,.0f}</td><td class="risk-val val-neutral">{currency}{r0_val:,.0f}</td><td class="risk-val val-calc">{currency}{r0_el:,.0f}</td><td class="risk-val val-risk">{r1_cnt:,.0f}</td><td class="risk-val val-risk">{currency}{r1_val:,.0f}</td><td class="risk-val val-calc">{currency}{r1_el:,.0f}</td></tr>')
            html.append('</tbody></table></div></div></div>')
            st.markdown("".join(html), unsafe_allow_html=True)
        else:
            if use_intersection: st.info("ℹ️ **No applicants match ALL selected risk criteria.**")
            else: st.info("ℹ️ **No applicants match ANY of the selected risk criteria.**")

        # --- D. GRID RENDER ---
        total_slots = len(current_cats)
        if len(current_cats) < 6: total_slots += 1
        cols = st.columns(3)
        
        for i in range(total_slots):
            col_idx = i % 3
            with cols[col_idx]:
                if i < len(current_cats):
                    category = current_cats[i]
                    
                    # 1. Stats (FIXED: Added 'Defaults' to columns to match agg)
                    group_stats = dff.groupby(category)['TARGET'].agg(['mean', 'count', 'sum']).reset_index()
                    group_stats.columns = [category, 'Default_Rate', 'Count', 'Defaults']
                    group_stats = group_stats.sort_values('Default_Rate', ascending=True) 
                    
                    # 2. Get Active Factors
                    active_factors = st.session_state['risk_sub_factors'].get(category, [])
                    
                    # 3. Metrics
                    if active_factors:
                        subset_stats = group_stats[group_stats[category].isin(active_factors)]
                        combined_rate = (subset_stats['Default_Rate'] * subset_stats['Count']).sum() / subset_stats['Count'].sum()
                        
                        group_count = subset_stats['Count'].sum()
                        share_of_portfolio = group_count / total_apps if total_apps > 0 else 0
                        risk_lift = combined_rate / global_mean if global_mean > 0 else 0
                        weighted_impact = combined_rate * share_of_portfolio
                        
                        factor_display = f"{len(active_factors)} selected"
                        if len(active_factors) <= 2: factor_display = ", ".join(active_factors)
                    else:
                        combined_rate = 0; risk_lift = 0; weighted_impact = 0; factor_display = "None"

                    if combined_rate < 0.05: story_color, icon = "#2ca02c", "🟢"
                    elif combined_rate < 0.10: story_color, icon = "#ff9800", "🟡"
                    else: story_color, icon = "#d62728", "🔴"

                    # 4. Narrative
                    if show_narrative:
                        lines = []
                        lines.append(f'<div class="narrative-box" style="border-left-color: {story_color}; font-family: \'Segoe UI\', sans-serif;">')
                        lines.append(f'<strong style="color: #eee;">{icon} Risk Factor(s)</strong><br>')
                        lines.append(f'<span style="font-size: 1.0em; color: #fff;">{factor_display}</span><br>')
                        lines.append(f'<span style="color: {story_color}; font-weight: bold; font-size: 1.2em;">Rate: {combined_rate:.1%}</span>')
                        lines.append(f'<div style="font-size: 0.9em; color: #aaa; margin-bottom: 8px;">Risk Lift: {risk_lift:.1f}x avg</div>')
                        lines.append(f'<div style="border-top: 1px solid rgba(255,255,255,0.1); padding-top: 6px; font-size: 0.85em; color: #888;">')
                        lines.append(f'Portfolio Impact: {weighted_impact:.2%} pts')
                        lines.append('</div></div>')
                        narrative_html = "".join(lines)

                    # 5. Gauge
                    if show_gauges:
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number+delta", value = combined_rate * 100, 
                            delta = {'reference': global_mean * 100, 'position': "top", 'relative': False, 'valueformat': ".1f"},
                            title = {'text': "Segment Risk", 'font': {'size': 14}}, number = {'suffix': "%", 'font': {'color': story_color}},
                            gauge = {'axis': {'range': [None, 50], 'tickwidth': 1}, 'bar': {'color': story_color}, 'bgcolor': "rgba(0,0,0,0)", 'borderwidth': 2, 'bordercolor': "gray", 'steps': [{'range': [0, 5], 'color': 'rgba(44, 160, 44, 0.1)'}, {'range': [5, 10], 'color': 'rgba(255, 152, 0, 0.1)'}, {'range': [10, 50], 'color': 'rgba(214, 39, 40, 0.1)'}], 'threshold': {'line': {'color': "white", 'width': 2}, 'thickness': 0.75, 'value': global_mean * 100}}
                        ))
                        fig_gauge.add_annotation(x=0.5, y=0.25, text="vs Portfolio Avg", showarrow=False, font=dict(size=10, color="#888"))
                        fig_gauge.update_layout(height=180, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})

                    # 6. Chart
                    if show_bars:
                        colors = ['#d62728' if x in active_factors else '#444' for x in group_stats[category]]
                        fig_bar = go.Figure()
                        fig_bar.add_trace(go.Bar(y=group_stats[category], x=[1]*len(group_stats), orientation='h', marker_color='rgba(255,255,255,0.05)', hoverinfo='skip'))
                        fig_bar.add_trace(go.Bar(y=group_stats[category], x=group_stats['Default_Rate'], orientation='h', marker=dict(color=colors, opacity=1), text=group_stats['Default_Rate'].apply(lambda x: f"{x:.1%}"), textposition='auto', hoverinfo='x+y', selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1))))
                        fig_bar.update_layout(title=dict(text=f"Breakdown: {category.replace('NAME_', '').title()}", font=dict(size=12)), barmode='overlay', showlegend=False, xaxis=dict(range=[0, max(0.2, group_stats['Default_Rate'].max()*1.2)], visible=False), yaxis=dict(showgrid=False), margin=dict(l=0, r=0, t=30, b=0), height=250, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', clickmode='event', dragmode=False, hovermode='y')

                    with st.container():
                        c_head, c_reset, c_del = st.columns([6, 1, 1])
                        c_head.markdown(f"#### {category.replace('NAME_', '').replace('_', ' ').title()}")
                        if c_reset.button("↺", key=f"reset_{category}", help="Reset to Max Risk"):
                            highest_risk_factor = group_stats.iloc[-1][category]
                            st.session_state['risk_sub_factors'][category] = [highest_risk_factor]
                            st.rerun()
                        if c_del.button("✕", key=f"del_{category}", help="Remove Chart"):
                            st.session_state['risk_selected_cats'].remove(category)
                            st.rerun()

                        if show_narrative: st.markdown(narrative_html, unsafe_allow_html=True)
                        if show_gauges: st.plotly_chart(fig_gauge, use_container_width=True, key=f"gauge_{category}")
                        if show_bars: st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{category}", on_select="rerun", selection_mode="points")
                        st.markdown("---")
                else:
                    with st.container():
                        st.markdown('<h4 style="color: #555;">➕ Add Risk Factor</h4>', unsafe_allow_html=True)
                        st.markdown('<p style="color: #555; font-size: 0.9em;">Compare another category:</p>', unsafe_allow_html=True)
                        for cat in available_cats:
                            btn_label = cat.replace('NAME_', '').replace('_', ' ').title()
                            if st.button(f"{btn_label}", key=f"add_{cat}"):
                                st.session_state['risk_selected_cats'].append(cat)
                                st.rerun()
                        st.markdown("---")

    # =========================================================
    # VIEW: HISTORICAL PIPELINE & TRENDS
    # =========================================================
    elif selected_sub_tab == "Historical Pipeline & Trends":
        st.subheader("Historical Pipeline & Trends")
        st.caption("Visualizing the flow of Approvals, Rejections, and Cancellations over time.")
        
        # 1. LOAD PREVIOUS APPLICATIONS
        try:
            prev_app_path = os.path.join("data", "previous_application (sample).csv")
            if os.path.exists(prev_app_path):
                pipe_df = pd.read_csv(prev_app_path)
            else:
                st.error("Pipeline Data not found.")
                return
        except Exception as e:
            st.error(f"Error: {e}")
            return
            
        # 2. SAMPLE
        if len(pipe_df) > max_points:
            viz_df = pipe_df.sample(max_points).copy()
            st.toast(f"Sampled to {max_points:,} points.", icon="⚡")
        else:
            viz_df = pipe_df.copy()
            
        # 3. DATE PREP & AXIS
        if show_dates:
            # Assume 0 = Today
            viz_df['Plot_X'] = pd.Timestamp.now() + pd.to_timedelta(viz_df['DAYS_DECISION'], unit='D')
            xaxis_title = "Decision Date"
            x_autorange = None 
            tick_fmt = "%b %Y"
            hover_fmt = "%{x|%b %Y}"
        else:
            viz_df['Plot_X'] = viz_df['DAYS_DECISION']
            xaxis_title = "Days Ago (Relative)"
            x_autorange = None 
            tick_fmt = None
            hover_fmt = "%{x} days ago"
        
        # 4. SUMMARY STATS
        total_apps = len(viz_df)
        total_apps_full = len(pipe_df)
        pct_shown = (total_apps / total_apps_full) * 100 if total_apps_full > 0 else 0
        
        # Define stats_df here (FIXED UnboundLocalError)
        stats_df = viz_df

        status_counts = viz_df['NAME_CONTRACT_STATUS'].value_counts()
        status_vals = viz_df.groupby('NAME_CONTRACT_STATUS')['AMT_APPLICATION'].sum()
        
        def get_stat(status):
            c = status_counts.get(status, 0)
            v = status_vals.get(status, 0)
            p = (c / total_apps) if total_apps > 0 else 0
            return c, v, p

        ap_c, ap_v, ap_p = get_stat("Approved")
        re_c, re_v, re_p = get_stat("Refused")
        ca_c, ca_v, ca_p = get_stat("Canceled")
        un_c, un_v, un_p = get_stat("Unused offer")

        st.markdown(f"""
        <div class="summary-card" style="padding: 15px; display: flex; justify-content: space-around; text-align: center;">
            <div><div style="font-size:0.8rem; color:#aaa;">SAMPLE VOLUME ({pct_shown:.1f}%)</div><div style="font-size:1.2rem; font-weight:bold; color:#fff;">{total_apps:,}</div></div>
            <div><div style="font-size:0.8rem; color:#2ca02c;">APPROVED</div><div style="font-size:1.2rem; font-weight:bold; color:#2ca02c;">{currency}{ap_v:,.0f}</div><div style="font-size:0.7rem;">{ap_c} apps ({ap_p:.1%})</div></div>
            <div><div style="font-size:0.8rem; color:#d62728;">DECLINED</div><div style="font-size:1.2rem; font-weight:bold; color:#d62728;">{currency}{re_v:,.0f}</div><div style="font-size:0.7rem;">{re_c} apps ({re_p:.1%})</div></div>
            <div><div style="font-size:0.8rem; color:#ffffff;">CANCELED</div><div style="font-size:1.2rem; font-weight:bold; color:#ffffff;">{currency}{ca_v:,.0f}</div><div style="font-size:0.7rem;">{ca_c} apps ({ca_p:.1%})</div></div>
            <div><div style="font-size:0.8rem; color:#ffd700;">UNUSED</div><div style="font-size:1.2rem; font-weight:bold; color:#ffd700;">{currency}{un_v:,.0f}</div><div style="font-size:0.7rem;">{un_c} apps ({un_p:.1%})</div></div>
        </div>
        """, unsafe_allow_html=True)

        # 5. CHART
        fig_pipe = go.Figure()
        fig_pipe.add_hline(y=0, line_width=1, line_color="#555")

        styles = {
            "Approved": {"y_sign": 1, "color": "#2ca02c", "symbol": "circle", "name": "✅ Approved"},
            "Canceled": {"y_sign": 1, "color": "#ffffff", "symbol": "circle", "name": "🚫 Canceled"},
            "Refused": {"y_sign": -1, "color": "#d62728", "symbol": "circle", "name": "❌ Declined"},
            "Unused offer": {"y_sign": -1, "color": "#ffd700", "symbol": "circle", "name": "🕸️ Unused"}
        }

        order = ["Refused", "Approved", "Canceled", "Unused offer"]
        
        # Use go.Scatter (SVG) if small sample, else go.Scattergl (WebGL)
        scatter_func = go.Scatter if total_apps < 8000 else go.Scattergl
        
        for status in order:
            subset = viz_df[viz_df['NAME_CONTRACT_STATUS'] == status]
            if not subset.empty:
                cfg = styles.get(status, {"y_sign": 1, "color": "white", "symbol": "circle", "name": status})
                y_values = subset['AMT_APPLICATION'] * cfg['y_sign']
                
                name_label = cfg['name']
                
                fig_pipe.add_trace(scatter_func(
                    x=subset['Plot_X'], y=y_values, mode='markers', name=name_label,
                    marker=dict(size=np.log1p(subset['AMT_APPLICATION']) * 0.8, color=cfg['color'], opacity=0.7, line=dict(width=0.5, color='black')),
                    text=subset['SK_ID_CURR'],
                    customdata=np.stack((subset['DAYS_DECISION'], subset.index), axis=-1),
                    hovertemplate=f"<b>{name_label}</b><br>ID: %{{text}}<br>Value: {currency}%{{y:,.0f}}<br>Date: {hover_fmt}<extra></extra>"
                ))
        
        # Add "Lollipop" Trace for Animation (Hidden Initially)
        fig_pipe.add_trace(go.Scatter(
            x=[], y=[], mode='markers+lines+text', name='Insight',
            marker=dict(size=25, color='rgba(0, 230, 255, 0.4)', symbol='circle', line=dict(width=3, color='#00e6ff')),
            line=dict(width=3, color='#00e6ff', dash='dot'),
            hoverinfo='skip', showlegend=False
        ))
        
        # --- ANIMATION FRAMES ---
        milestones = []
        try:
            daily = viz_df.groupby(['DAYS_DECISION'])['AMT_APPLICATION'].sum().reset_index()
            # 1. Peak Volume
            if not daily.empty:
                max_vol_idx = daily['AMT_APPLICATION'].idxmax()
                m1 = daily.iloc[max_vol_idx]
                milestones.append({"day": m1['DAYS_DECISION'], "val": m1['AMT_APPLICATION'], "label": "Peak Volume", "y": m1['AMT_APPLICATION'], "color": "#00e6ff"})
                
                # 2. Lowest Volume
                daily_nz = daily[daily['AMT_APPLICATION'] > 0]
                if not daily_nz.empty:
                    min_vol_idx = daily_nz['AMT_APPLICATION'].idxmin()
                    m_min = daily_nz.loc[min_vol_idx]
                    milestones.append({"day": m_min['DAYS_DECISION'], "val": m_min['AMT_APPLICATION'], "label": "Lowest Volume", "y": m_min['AMT_APPLICATION'], "color": "#888"})
            
            # 3. Max Approvals
            app_daily = viz_df[viz_df['NAME_CONTRACT_STATUS']=='Approved'].groupby('DAYS_DECISION')['AMT_APPLICATION'].sum()
            if not app_daily.empty:
                m2_day = app_daily.idxmax()
                m2_val = app_daily.max()
                milestones.append({"day": m2_day, "val": m2_val, "label": "Highest Approvals", "y": m2_val, "color": "#2ca02c"})

            # 4. Max Declines
            ref_daily = viz_df[viz_df['NAME_CONTRACT_STATUS']=='Refused'].groupby('DAYS_DECISION')['AMT_APPLICATION'].sum()
            if not ref_daily.empty:
                m3_day = ref_daily.idxmax()
                m3_val = ref_daily.max()
                milestones.append({"day": m3_day, "val": m3_val, "label": "Highest Declines", "y": -m3_val, "color": "#d62728"})
            
            # 5. Max Cancellations
            can_daily = viz_df[viz_df['NAME_CONTRACT_STATUS']=='Canceled'].groupby('DAYS_DECISION')['AMT_APPLICATION'].sum()
            if not can_daily.empty:
                m4_day = can_daily.idxmax()
                m4_val = can_daily.max()
                milestones.append({"day": m4_day, "val": m4_val, "label": "Highest Cancellations", "y": m4_val, "color": "#ffffff"})

            # 6. Max Unused
            un_daily = viz_df[viz_df['NAME_CONTRACT_STATUS']=='Unused offer'].groupby('DAYS_DECISION')['AMT_APPLICATION'].sum()
            if not un_daily.empty:
                m5_day = un_daily.idxmax()
                m5_val = un_daily.max()
                milestones.append({"day": m5_day, "val": m5_val, "label": "Highest Unused", "y": -m5_val, "color": "#ffd700"})
                
            milestones.sort(key=lambda x: x['day'])
        except: pass

        frames = []
        highlight_idx = len(fig_pipe.data) - 1
        
        # Fixed Lollipop Height (25% of Max Y)
        y_max_scale = viz_df['AMT_APPLICATION'].quantile(0.95) if not viz_df.empty else 100000
        lollipop_h = y_max_scale * 0.25
        
        for m in milestones:
            if show_dates: x_pos = pd.Timestamp.now() + pd.to_timedelta(m['day'], unit='D')
            else: x_pos = m['day']
            
            # Use fixed height relative to axis, sign determines direction
            is_pos = m['y'] > 0
            stick_y = lollipop_h if is_pos else -lollipop_h
            
            frames.append(go.Frame(
                data=[go.Scatter(
                    x=[x_pos, x_pos], 
                    y=[0, stick_y], 
                    mode='lines+markers',
                    marker=dict(size=14, color=m['color'], symbol='circle', line=dict(width=2, color='white')),
                    line=dict(width=3, color=m['color'], dash='dot')
                )],
                traces=[highlight_idx],
                layout=dict(
                    annotations=[dict(
                        x=x_pos, y=stick_y, yref="y", # Use Data Coordinates
                        text=f"<span style='font-family:Consolas; color:#eee; background-color:rgba(0,0,0,0.8); padding:6px; border-radius:4px;'>{m['label']}<br>{currency}{m['val']:,.0f}</span>",
                        showarrow=True, arrowhead=2, ax=0, ay=-40 if is_pos else 40,
                        font=dict(size=12, color="white"),
                        arrowcolor=m['color']
                    )]
                )
            ))

        fig_pipe.frames = frames

        fig_pipe.update_layout(
            title=f"Application Decisions Timeline (Sample: {total_apps:,} / {pct_shown:.1f}% of {total_apps_full:,})",
            xaxis=dict(title=xaxis_title, autorange=x_autorange, tickformat=tick_fmt),
            yaxis=dict(title=f"Application Amount ({currency})", tickprefix=currency),
            height=750, 
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=60, b=80), 
            dragmode='pan',
            updatemenus=[dict(
                type="buttons", 
                direction="left",
                showactive=False, 
                x=0.01, y=-0.15,
                bgcolor='rgba(0,0,0,0)', bordercolor='#eee', borderwidth=1, font=dict(color="#eee"),
                buttons=[
                    dict(label="▶ Show Insights", method="animate", args=[None, dict(frame=dict(duration=2500, redraw=True), fromcurrent=True)]),
                    dict(label="⏸ Stop", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])
                ]
            )]
        )
        
        fig_pipe.update_xaxes(rangeslider_visible=False) 
        
        selection = st.plotly_chart(fig_pipe, use_container_width=True, on_select="rerun", selection_mode="points", key="pipeline_chart")
    
        if selection and selection["selection"]["points"]:
            point = selection["selection"]["points"][0]
            try:
                # Check curveNumber to avoid clicking the highlight line
                if point['curveNumber'] != highlight_idx:
                    row_idx = point['customdata'][1]
                    applicant_data = viz_df.loc[row_idx]
                    show_applicant_popup(applicant_data, currency)
            except Exception as e:
                pass

        # 6. TREND NARRATIVE (FIXED: Uses stats_df and includes unused_rate)
        total_v = stats_df['AMT_APPLICATION'].sum()
        app_rate = ap_v / total_v if total_v > 0 else 0
        ref_rate = re_v / total_v if total_v > 0 else 0
        unused_rate = un_v / total_v if total_v > 0 else 0
        
        avg_app = ap_v / ap_c if ap_c > 0 else 0
        avg_ref = re_v / re_c if re_c > 0 else 0

        cutoff = stats_df['DAYS_DECISION'].quantile(0.8)
        recent = stats_df[stats_df['DAYS_DECISION'] > cutoff]
        rec_app_rate = 0
        if not recent.empty:
            rec_app_rate = recent[recent['NAME_CONTRACT_STATUS']=='Approved']['AMT_APPLICATION'].sum() / recent['AMT_APPLICATION'].sum()
        
        trend_icon = "↗️" if rec_app_rate > app_rate else "↘️"
        
        st.info(f"""
        **Pipeline Insights (Based on {total_apps:,} Samples / {pct_shown:.1f}% of {total_apps_full:,}):**
        - **Approval Rate:** {app_rate:.1%} of total value requested.
        - **Rejection Rate:** {ref_rate:.1%} of value was rejected to mitigate risk.
        - **Unused Offers:** {unused_rate:.1%} of approved credit was not activated by clients.
        - **Avg Ticket Size:** Approved loans avg {currency}{avg_app:,.0f} vs Refused avg {currency}{avg_ref:,.0f}.
        - **Recent Trend:** Approval rate is currently **{rec_app_rate:.1%}** (vs {app_rate:.1%} historical avg) {trend_icon}.
        """)

    # =========================================================
    # VIEW: FEATURE INFLUENCE
    # =========================================================
    elif selected_sub_tab == "Feature Influence":
        st.header("🔍 Feature Influence & Interaction")
        st.markdown("Analyze which factors drive default risk and how they combine.")
        
        subtab_a, subtab_b, subtab_c = st.tabs(["Drivers", "Risk Matrix", "Trend Analysis"])
        
        with subtab_a:
            st.subheader("Top Risk Drivers")
            numeric_df = dff.select_dtypes(include=[np.number])
            if 'TARGET' in numeric_df.columns:
                corr = numeric_df.corrwith(dff['TARGET']).sort_values()
                top_neg = corr.head(10)
                top_pos = corr.tail(11).drop('TARGET', errors='ignore')
                corr_data = pd.concat([top_neg, top_pos])
                fig_corr = go.Figure()
                fig_corr.add_trace(go.Bar(
                    y=corr_data.index, x=corr_data.values, orientation='h',
                    marker=dict(color=corr_data.values, colorscale='RdBu_r', cmid=0)
                ))
                fig_corr.update_layout(title="Feature Correlation with Default", xaxis_title="Correlation Coefficient", height=600, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_corr, use_container_width=True)
        
        with subtab_b:
            st.subheader("⚠️ Risk Matrix: Exploring Combinations")
            col_x, col_y = st.columns(2)
            num_cols = dff.select_dtypes(include=[np.number]).columns.tolist()
            default_x = 'AMT_INCOME_TOTAL' if 'AMT_INCOME_TOTAL' in num_cols else num_cols[0]
            default_y = 'EXT_SOURCE_2' if 'EXT_SOURCE_2' in num_cols else num_cols[1]
            x_feat = col_x.selectbox("X-Axis Feature", num_cols, index=num_cols.index(default_x))
            y_feat = col_y.selectbox("Y-Axis Feature", num_cols, index=num_cols.index(default_y))
            if x_feat and y_feat:
                try:
                    dff['X_Bin'] = pd.qcut(dff[x_feat], q=5, duplicates='drop').astype(str)
                    dff['Y_Bin'] = pd.qcut(dff[y_feat], q=5, duplicates='drop').astype(str)
                    metric_val = 'TARGET' if matrix_metric == "Default Rate" else 'AMT_CREDIT'
                    heatmap_data = dff.pivot_table(index='Y_Bin', columns='X_Bin', values=metric_val, aggfunc='mean')
                    fig_matrix = px.imshow(heatmap_data, labels=dict(x=x_feat, y=y_feat, color="Default Rate"), color_continuous_scale="Reds", text_auto='.2f', title=f"Risk Matrix: {x_feat} vs {y_feat}")
                    fig_matrix.update_layout(height=500)
                    st.plotly_chart(fig_matrix, use_container_width=True)
                except Exception as e: st.error(f"Error: {e}")

        with subtab_c:
            st.subheader("Feature Trend Analysis (Bad Rate)")
            trend_feat = st.selectbox("Select Feature to Analyze", num_cols, index=num_cols.index('DAYS_BIRTH') if 'DAYS_BIRTH' in num_cols else 0)
            if trend_feat:
                try:
                    dff['Trend_Bin'] = pd.qcut(dff[trend_feat], q=10, duplicates='drop')
                    trend_data = dff.groupby('Trend_Bin')[['TARGET']].mean().reset_index()
                    trend_data['Trend_Bin'] = trend_data['Trend_Bin'].astype(str)
                    fig_trend = px.line(trend_data, x='Trend_Bin', y='TARGET', markers=True, title=f"Default Rate by {trend_feat}")
                    fig_trend.update_traces(line_color='#d62728', line_width=3)
                    fig_trend.add_hline(y=dff['TARGET'].mean(), line_dash="dot", annotation_text="Portfolio Avg", line_color="gray")
                    if show_trend_line: st.plotly_chart(fig_trend, use_container_width=True)
                except: st.warning("Feature may be categorical or have too few unique values.")