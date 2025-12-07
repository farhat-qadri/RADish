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
from modules import data_engine

def show(df):
    st.header("Risk Drivers & Exploration")
    st.markdown("Deep dive into correlations and individual customer journeys.")
    
    # ---------------------------------------------------------
    # PB 20251206 CONFIG & CURRENCY
    # ---------------------------------------------------------
    # Get Currency Symbol from Preferences
    currency = data_engine.get_currency_symbol()

    # ---------------------------------------------------------
    # PB 20251206 TABS
    # ---------------------------------------------------------
    # Simplified to focus on Drivers and Journey. PCA/Social are now in "Deep Dive".
    tab1, tab2 = st.tabs(["Correlation Field", "Customer Journey (Timeline)"])
    
    # =========================================================
    # PB 20251204 TAB 1: CORRELATION FIELD
    # =========================================================
    with tab1:
        st.subheader("Interactive Correlation Matrix")
        
        # Filter numeric columns
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Default selections
        defaults = ['TARGET', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'EXT_SOURCE_2']
        defaults = [c for c in defaults if c in num_cols]
        
        selected_cols = st.multiselect("Select Features to Correlate", num_cols, default=defaults)
        
        if selected_cols:
            corr_matrix = df[selected_cols].corr()
            fig_corr = px.imshow(
                corr_matrix, 
                text_auto=".2f", 
                color_continuous_scale="RdBu_r", 
                zmin=-1, zmax=1, 
                aspect="auto", 
                title="Correlation Heatmap"
            )
            st.plotly_chart(fig_corr, use_container_width=True)

    # =========================================================
    # PB 20251205 TAB 2: CUSTOMER JOURNEY (GANTT + FLAGS)
    # =========================================================
    with tab2:
        st.subheader("Loan Lifecycle Timeline (Gantt View)")
        st.caption("Visualize the temporal history of all applications with event status icons.")
        
        # 1. SMART DROPDOWN
        # Sample for performance (or use full list if optimized)
        sample_df = df.head(200) # Increased sample size
        
        id_options = {}
        for idx, row in sample_df.iterrows():
            sk_id = row['SK_ID_CURR']
            # Count Apps
            total_apps = 1 + int(row.get('PREV_APP_COUNT', 0))
            
            # Count Flags (Car + Realty + Previous Refusals)
            flag_count = 0
            if row.get('FLAG_OWN_CAR') == 'Y': flag_count += 1
            if row.get('FLAG_OWN_REALTY') == 'Y': flag_count += 1
            flag_count += int(row.get('PREV_REFUSALS', 0))
            
            id_options[sk_id] = f"{sk_id} [{total_apps:03d} Apps | {flag_count:02d} Flags]"
            
        selected_id = st.selectbox(
            "Select Applicant (ID [Count | Flags])", 
            options=list(id_options.keys()),
            format_func=lambda x: id_options.get(x, str(x))
        )
        
        if selected_id:
            # 2. GET DATA
            history = data_engine.get_applicant_history(selected_id)
            current_app = df[df['SK_ID_CURR'] == selected_id].iloc[0]
            
            # --- ICON MAPPER ---
            def get_icon(status, event_type):
                s = str(status).lower()
                e = str(event_type).lower()
                if 'approved' in s: return "✅"
                if 'refused' in s: return "❌"
                if 'canceled' in s: return "🚫"
                if 'unused' in s: return "🕸️"
                if 'processing' in s: return "⏳"
                if 'disbursement' in e: return "💰"
                if 'due' in e: return "📅"
                if 'termination' in e: return "🏁"
                if 'application' in e: return "📝"
                return "🔹"

            # --- BUILD TIMELINE EVENTS ---
            timeline_events = []

            # A. Current Application (T=0)
            curr_flags = []
            if current_app.get('FLAG_OWN_CAR') == 'Y': curr_flags.append('🚗 Car')
            if current_app.get('FLAG_OWN_REALTY') == 'Y': curr_flags.append('🏠 Realty')
            flag_desc = f" | Flags: {', '.join(curr_flags)}" if curr_flags else ""

            curr_label = f"Current: {current_app['NAME_CONTRACT_TYPE']} ({currency}{current_app['AMT_CREDIT']:,.0f})"
            timeline_events.append({
                'Loan_ID': curr_label,
                'Days_Raw': 0,
                'Event': 'Application',
                'Status': 'Processing',
                'Amount': current_app['AMT_CREDIT'],
                'Desc': f'Target Analysis Ongoing{flag_desc}',
                'Icon': '⏳'
            })

            # B. Historical Applications
            if not history.empty:
                for _, row in history.iterrows():
                    prev_label = f"Prev ({row['NAME_CONTRACT_STATUS']}): {row['NAME_CONTRACT_TYPE']} ({currency}{row['AMT_CREDIT']:,.0f})"
                    
                    # History Flags
                    prev_flags = []
                    if row.get('NFLAG_INSURED_ON_APPROVAL') == 1: prev_flags.append('🛡️ Insured')
                    if row.get('NFLAG_LAST_APPL_IN_DAY') == 0: prev_flags.append('⚠️ Not Last App')
                    p_flag_desc = f" | {', '.join(prev_flags)}" if prev_flags else ""

                    # 1. Decision Event
                    timeline_events.append({
                        'Loan_ID': prev_label,
                        'Days_Raw': row['DAYS_DECISION'],
                        'Event': 'Decision',
                        'Status': row['NAME_CONTRACT_STATUS'],
                        'Amount': row['AMT_CREDIT'],
                        'Desc': f"Outcome: {row['NAME_CONTRACT_STATUS']}{p_flag_desc}",
                        'Icon': get_icon(row['NAME_CONTRACT_STATUS'], 'Decision')
                    })
                    
                    # 2. Lifecycle Events (Unpivoting)
                    def add_event(col, name, status):
                        if col in row and pd.notnull(row[col]) and row[col] < 0 and row[col] > -36000:
                             timeline_events.append({
                                'Loan_ID': prev_label,
                                'Days_Raw': row[col],
                                'Event': name,
                                'Status': status,
                                'Amount': row.get('AMT_ANNUITY', 0),
                                'Desc': name,
                                'Icon': get_icon(status, name)
                            })

                    add_event('DAYS_FIRST_DRAWING', 'Disbursement', 'Active')
                    add_event('DAYS_FIRST_DUE', 'First Due', 'Active')
                    add_event('DAYS_TERMINATION', 'Termination', 'Closed')

            full_timeline = pd.DataFrame(timeline_events)
            
            # SORTING: Loan Order (Current on Top)
            loan_order = full_timeline.groupby('Loan_ID')['Days_Raw'].max().sort_values(ascending=True).index.tolist()
            
            # --- PLOTLY GANTT CHART ---
            fig = go.Figure()
            
            color_map = {
                'Approved': '#2ca02c', 'Refused': '#d62728', 'Canceled': '#ff9800', 
                'Unused offer': '#7f7f7f', 'Processing': '#17a2b8', 'Active': '#9467bd', 
                'Closed': '#8c564b'
            }

            for loan in loan_order:
                loan_data = full_timeline[full_timeline['Loan_ID'] == loan]
                
                # 1. Swimlane Line
                min_day = loan_data['Days_Raw'].min()
                max_day = loan_data['Days_Raw'].max()
                
                fig.add_trace(go.Scatter(
                    x=[min_day, max_day], y=[loan, loan],
                    mode='lines',
                    line=dict(color='#444', width=2),
                    hoverinfo='skip', showlegend=False
                ))
                
                # 2. Markers with Icons
                sizes = []
                colors = []
                icons = []
                hover_texts = []
                
                for _, row in loan_data.iterrows():
                    # Bubble Size (Amount)
                    base_size = np.log1p(row['Amount'] if row['Amount'] else 0) * 1.8
                    sizes.append(max(20, base_size)) # Minimum size for Icon visibility
                    
                    colors.append(color_map.get(row['Status'], '#888'))
                    icons.append(row['Icon'])
                    
                    days_ago_str = f"{abs(int(row['Days_Raw']))} days ago" if row['Days_Raw'] != 0 else "Today"
                    hover_texts.append(
                        f"<b>{row['Event']}</b> {row['Icon']}<br>"
                        f"{row['Desc']}<br>"
                        f"📅 {days_ago_str}<br>"
                        f"💰 {currency}{row['Amount']:,.0f}"
                    )

                fig.add_trace(go.Scatter(
                    x=loan_data['Days_Raw'],
                    y=[loan] * len(loan_data),
                    mode='markers+text', # Show Marker AND Text (Icon)
                    marker=dict(
                        size=sizes, 
                        color=colors, 
                        line=dict(width=1, color='white'), 
                        opacity=0.3 # Semi-transparent bubble so Icon pops
                    ),
                    text=icons, # The Emoji
                    textposition="middle center",
                    textfont=dict(size=14, color="white"), # Icon style
                    hovertext=hover_texts,
                    hoverinfo='text',
                    name=loan,
                    showlegend=False
                ))

            # 3. Pulsing Halo (Critical)
            critical = full_timeline[full_timeline['Status'].isin(['Refused', 'Processing'])]
            if not critical.empty:
                fig.add_trace(go.Scatter(
                    x=critical['Days_Raw'], y=critical['Loan_ID'],
                    mode='markers',
                    marker=dict(size=45, color='red', opacity=0.1), # Faint Halo
                    hoverinfo='skip', showlegend=False
                ))

            # 4. Compact Layout
            fig.update_layout(
                title=f"Customer Journey: {id_options[selected_id]}",
                xaxis=dict(title="Timeline (Time Flows Left to Right)", showgrid=True, gridcolor='#333'),
                yaxis=dict(title="", type='category', categoryorder='array', categoryarray=loan_order),
                # REDUCED HEIGHT FACTOR from 30 to 25 per row
                height=300 + (len(loan_order) * 25), 
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # --- TABLE (Newest First) ---
            st.markdown("##### 📜 Event Log")
            table_df = full_timeline.sort_values('Days_Raw', ascending=False).copy()
            table_df['Days Ago'] = table_df['Days_Raw'].apply(lambda x: f"{abs(int(x))}" if x != 0 else "0 (Today)")
            table_df = table_df[['Days Ago', 'Icon', 'Loan_ID', 'Event', 'Status', 'Desc', 'Amount']]
            
            st.dataframe(
                table_df.style.map(lambda x: f"color: {color_map.get(x, 'white')}", subset=['Status'])
                .format({'Amount': f"{currency}{{:.0f}}"}),
                use_container_width=True
            )