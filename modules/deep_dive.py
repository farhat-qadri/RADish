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

#   20251207 Added Deep Dive

import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def show(df):
    # ---------------------------------------------------------
    # PB 20251206 SETUP SIDEBAR 
    # ---------------------------------------------------------
    st.sidebar.markdown('<div class="sidebar-section">1. Global Filter</div>', unsafe_allow_html=True)
    contract_filter = st.sidebar.multiselect(
        "Contract Type", 
        df['NAME_CONTRACT_TYPE'].unique(), 
        default=df['NAME_CONTRACT_TYPE'].unique(),
        label_visibility="collapsed"
    )
    
    if not contract_filter:
        st.warning("Please select a contract type.")
        return
    
    dff = df[df['NAME_CONTRACT_TYPE'].isin(contract_filter)]

    # ---------------------------------------------------------
    # PB 20251205 SHOW TABS
    # ---------------------------------------------------------
    st.title("🔬 Deep Dive Analysis")
    
    tab_pca, tab_social = st.tabs(["PCA Clusters (Risk Neighborhoods)", "Social Network Analysis"])

    # =========================================================
    # TAB 1: PCA CLUSTERS
    # =========================================================
    with tab_pca:
        st.markdown("Explore **Risk Neighborhoods** using advanced dimensionality reduction (PCA).")
        
        # --- FEATURE SELECTION SIDEBAR ---
        st.sidebar.markdown('<div class="sidebar-section">2. PCA Features</div>', unsafe_allow_html=True)
        
        numeric_cols = dff.select_dtypes(include=[np.number]).columns.tolist()
        
        # Updated Presets (No Icons)
        presets = {
            "Financial": ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE'],
            "Ext Credit Scores": ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'REGION_RATING_CLIENT'],
            "Demographics": ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'CNT_CHILDREN', 'OWN_CAR_AGE'],
            "Balanced": ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'DAYS_BIRTH', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
        }

        # Initialize State for Features
        if 'dd_features' not in st.session_state:
            st.session_state['dd_features'] = presets["Balanced"]

        # Callback function for Dropdown
        def on_preset_change():
            selection = st.session_state['pca_preset_selector']
            if selection != "Custom":
                # Update the features based on preset
                st.session_state['dd_features'] = [c for c in presets[selection] if c in numeric_cols]

        # Dropdown Selector
        st.sidebar.selectbox(
            "Quick Selectors",
            options=["Balanced", "Financial", "Ext Credit Scores", "Demographics", "Custom"],
            key='pca_preset_selector',
            on_change=on_preset_change
        )

        st.sidebar.markdown("---")

        # Multiselect (Linked to State)
        selected_features = st.sidebar.multiselect(
            "Custom Selection", 
            options=numeric_cols, 
            default=st.session_state['dd_features'],
            key='dd_features' # Binding directly allows 2-way sync with callback
        )

        # --- SIDEBAR: DISPLAY OPTIONS ---
        st.sidebar.markdown('<div class="sidebar-section">3. Display</div>', unsafe_allow_html=True)
        show_accepted = st.sidebar.toggle("Show Accepted (Grey)", value=True)
        show_rejected = st.sidebar.toggle("Show Rejected (Red)", value=True)

        if len(selected_features) < 2:
            st.warning("⚠️ Please select at least 2 features from the sidebar.")
        else:
            with st.spinner(f"Projecting {len(selected_features)} dimensions..."):
                # Prep Data
                meta_cols = ['TARGET', 'SK_ID_CURR', 'NAME_CONTRACT_TYPE', 'AMT_CREDIT', 'AMT_INCOME_TOTAL']
                meta_cols = [c for c in meta_cols if c in dff.columns]
                cols_needed = list(set(selected_features + meta_cols))
                
                subset = dff[cols_needed].dropna(subset=selected_features)
                if len(subset) > 2000: subset = subset.sample(2000)
                
                if subset.empty:
                    st.error("Not enough data points.")
                else:
                    # PCA
                    scaler = StandardScaler()
                    x_scaled = scaler.fit_transform(subset[selected_features])
                    pca = PCA(n_components=2)
                    components = pca.fit_transform(x_scaled)
                    
                    subset['PC1'] = components[:,0]
                    subset['PC2'] = components[:,1]
                    
                    # Interpret Loadings
                    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=selected_features)
                    pc1_driver = loadings['PC1'].abs().idxmax()
                    pc2_driver = loadings['PC2'].abs().idxmax()

                    # --- FILTER VISIBILITY ---
                    mask_accepted = (subset['TARGET'] == 0) & show_accepted
                    mask_rejected = (subset['TARGET'] == 1) & show_rejected
                    
                    subset['Risk Status'] = subset['TARGET'].map({0: 'Accepted', 1: 'Rejected'})
                    viz_subset = subset[mask_accepted | mask_rejected].copy()
                    
                    # Calculate Global Bounds for Fixed Scale
                    x_min, x_max = subset['PC1'].min(), subset['PC1'].max()
                    y_min, y_max = subset['PC2'].min(), subset['PC2'].max()
                    pad_x = (x_max - x_min) * 0.1
                    pad_y = (y_max - y_min) * 0.1
                    range_x = [x_min - pad_x, x_max + pad_x]
                    range_y = [y_min - pad_y, y_max + pad_y]
                    
                    if viz_subset.empty:
                        st.info("No data points visible. Check display toggles.")
                    else:
                        # Plot
                        size_col = 'AMT_CREDIT' if 'AMT_CREDIT' in viz_subset.columns else None
                        
                        fig = px.scatter(
                            viz_subset, x='PC1', y='PC2', 
                            color='Risk Status',
                            color_discrete_map={'Accepted': 'lightgrey', 'Rejected': '#d62728'},
                            size=size_col, 
                            title=f"Risk Clusters (Driven by {pc1_driver} & {pc2_driver})",
                            labels={'PC1': f"Axis 1 ({pc1_driver})", 'PC2': f"Axis 2 ({pc2_driver})"},
                            hover_data=['SK_ID_CURR'] + selected_features[:5],
                            opacity=0.8
                        )
                        fig.update_xaxes(range=range_x)
                        fig.update_yaxes(range=range_y)
                        fig.update_layout(height=600, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig, use_container_width=True)

                    # Insights
                    c_txt, c_tbl = st.columns([1, 1])
                    with c_txt:
                        st.subheader("📝 Assessment")
                        risk_mean = subset[subset['TARGET']==1][selected_features].mean()
                        safe_mean = subset[subset['TARGET']==0][selected_features].mean()
                        
                        if not risk_mean.empty:
                            diffs = ((risk_mean - safe_mean) / safe_mean) * 100
                            top_diffs = diffs.abs().sort_values(ascending=False).head(3)
                            narrative = [f"- **{feat}** is **{abs(val):.1f}% {'higher' if val>0 else 'lower'}**" for feat, val in top_diffs.items()]
                            st.info(f"**Red Zone Profile:**\nHigh-risk applicants typically differ by:\n" + "\n".join(narrative))
                    
                    with c_tbl:
                        st.subheader("🚨 Details")
                        high_risk = subset[subset['TARGET'] == 1].head(50)
                        disp_cols = ['SK_ID_CURR'] + selected_features[:3]
                        if 'AMT_CREDIT' in subset.columns and 'AMT_CREDIT' not in disp_cols: disp_cols.append('AMT_CREDIT')
                        st.dataframe(high_risk[disp_cols], use_container_width=True, height=200)

    # =========================================================
    # PB20251207 SOCIAL NETWORK ANALYSIS (Peers default)
    # =========================================================
    with tab_social:
        st.subheader("Social Surroundings Analysis")
        st.caption("Analysis of applicant's social circle observations (30 Days Past Due).")
        
        if 'OBS_30_CNT_SOCIAL_CIRCLE' in dff.columns and 'DEF_30_CNT_SOCIAL_CIRCLE' in dff.columns:
            col_soc1, col_soc2 = st.columns([3, 1])
            with col_soc1:
                social_df = dff[['OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'TARGET', 'AMT_INCOME_TOTAL']].dropna().sample(min(2000, len(dff)))
                social_df['Jitter_X'] = social_df['OBS_30_CNT_SOCIAL_CIRCLE'] + np.random.normal(0, 0.2, len(social_df))
                social_df['Jitter_Y'] = social_df['DEF_30_CNT_SOCIAL_CIRCLE'] + np.random.normal(0, 0.2, len(social_df))
                
                fig_social = px.scatter(
                    social_df, x='Jitter_X', y='Jitter_Y', color='TARGET',
                    color_continuous_scale=["lightgrey", "#d62728"], size='AMT_INCOME_TOTAL',
                    labels={'Jitter_X': 'Observed Social Circle Size', 'Jitter_Y': 'Defaulters in Circle', 'TARGET': 'Risk'},
                    title="Social Risk Map"
                )
                fig_social.add_shape(type="line", x0=0, y0=0, x1=20, y1=4, line=dict(color="Red", width=2, dash="dash"))
                fig_social.update_layout(height=500, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_social, use_container_width=True)
            
            with col_soc2:
                st.markdown("#### 📊 Insights")
                avg_circle = dff['OBS_30_CNT_SOCIAL_CIRCLE'].mean()
                avg_def = dff['DEF_30_CNT_SOCIAL_CIRCLE'].mean()
                st.metric("Avg Social Circle", f"{avg_circle:.1f}")
                st.metric("Avg Defaulters", f"{avg_def:.2f}")
                
                risky_circles = dff[dff['DEF_30_CNT_SOCIAL_CIRCLE'] > 0]
                if not risky_circles.empty:
                    risk_prob = risky_circles['TARGET'].mean()
                    global_prob = dff['TARGET'].mean()
                    lift = risk_prob / global_prob if global_prob > 0 else 0
                    st.metric("Risk Lift", f"{lift:.1f}x", help="Higher risk if friends default.")
                st.info("Y-Axis = # of defaults in social circle.")
        else:
            st.warning("Social Circle data not available.")