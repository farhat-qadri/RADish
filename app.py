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
from streamlit_option_menu import option_menu
from modules import data_engine, landscape, drivers, assessment, deep_dive
import os
import pandas as pd

# ---------------------------------------------------------
# PB 20251204 INIT & CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="RADISH | Risk Assessment Dashboard", 
    page_icon="assets/favicon.ico",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# PB 20251205 LOAD CSS
# ---------------------------------------------------------
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

st.markdown("""
    <style>
        /* 1. LAYOUT & SPACING */
        .block-container {
            padding-top: 1rem !important; 
            padding-bottom: 3rem !important;
        }
        [data-testid="stHeader"] { background-color: transparent; }
        [data-testid="stSidebar"] { padding-top: 1rem; }

        /* 2. INDUSTRIAL CARD STYLING */
        /* Targets the st.container(border=True) elements */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.08); /* Crisp border */
            border-radius: 8px; /* Tighter corners */
            padding: 20px;
            height: 100%;        
            min-height: 550px;   /* FIXED: Increased to 550px to match tallest card */
            display: flex;
            flex-direction: column;
            transition: border-color 0.3s ease;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: rgba(23, 162, 184, 0.4); /* Teal glow on hover */
        }
        
        /* 3. CUSTOM STATUS PILLS */
        .status-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 12px;
            margin-bottom: 8px;
            background: rgba(0,0,0,0.2);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 4px;
            font-family: 'Consolas', 'Monaco', monospace; 
        }
        .status-left { display: flex; flex-direction: column; }
        .status-name { font-size: 0.85rem; color: #eee; font-weight: 500; }
        .status-meta { font-size: 0.7rem; color: #666; margin-top: 2px; }
        .status-pill {
            font-size: 0.7rem; padding: 3px 8px; border-radius: 12px;
            font-weight: 600; text-transform: uppercase; letter-spacing: 1px;
            min-width: 70px; text-align: center;
        }
        .status-ok { color: #4caf50; background: rgba(76, 175, 80, 0.1); border: 1px solid rgba(76, 175, 80, 0.2); }
        .status-miss { color: #d62728; background: rgba(214, 39, 40, 0.1); border: 1px solid rgba(214, 39, 40, 0.2); }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# PB 20251206 SIDEBAR BRANDING
# ---------------------------------------------------------
with st.sidebar:
    if os.path.exists("assets/logo.png"):
        st.image("assets/logo.png", width=180)
    
    st.markdown("""
        <p style='color: #888; font-size: 0.9em; margin-top: 5px; font-weight: 600;'>Risk Assessment Dashboard</p>
        <hr style='margin-top: 10px; margin-bottom: 20px; border-color: #333;'>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# PB 20251204 TOP NAVIGATION BAR
# ---------------------------------------------------------
selected = option_menu(
    menu_title=None,
    options=["Data Sources", "Risk Landscape", "Risk Drivers", "Deep Dive", "Risk Assessment"],
    default_index=0, 
    orientation="horizontal",
    styles={
        "container": {"padding": "5px !important", "background-color": "transparent", "margin": "0"},
        "nav-link": {
            "font-size": "13px", "text-align": "center", "margin":"0px 5px", "padding": "10px 10px",
            "border-radius": "8px", "border": "1px solid rgba(255,255,255,0.05)"
        },
        "nav-link-selected": {
            "background-color": "#17a2b8", "font-weight": "600", "color": "white", "border": "1px solid #17a2b8"
        }, 
    }
)

# ---------------------------------------------------------
# PB 20251205 MAIN MODULE CONTROLLER
# ---------------------------------------------------------

if selected == "Data Sources":
    st.title("Data Management")
    
    settings = data_engine.load_settings()
    last_update = settings.get("last_refresh", "Never")
    
    c_meta1, c_meta2, _ = st.columns([1, 1, 4])
    c_meta1.caption(f"System: v1.0 | Stable")
    c_meta2.caption(f"Last Build: {last_update}")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    # --- CARD 1: FILE STATUS ---
    with col1:
        with st.container(border=True):
            st.subheader("File Status")
            st.caption("Monitoring input directory")
            st.write("---")
            
            status = data_engine.check_data_status()
            file_map = {
                "Raw - Application": "application_data (sample).csv",
                "Raw - Previous App": "previous_application (sample).csv",
                "Cleaned - Merged": "radish_merged_data.csv"
            }
            
            html_status = []
            for name, state in status.items():
                meta_info = "N/A"
                if "Available" in state:
                    css_class, label, icon = "status-ok", "ONLINE", "●"
                    fname = file_map.get(name)
                    try:
                        if fname and os.path.exists(os.path.join("data", fname)):
                            df_temp = pd.read_csv(os.path.join("data", fname), nrows=5)
                            full_len = sum(1 for row in open(os.path.join("data", fname), 'r')) - 1
                            meta_info = f"{full_len:,} x {df_temp.shape[1]} cols"
                    except:
                        meta_info = "Ready"
                else:
                    css_class, label, icon = "status-miss", "MISSING", "○"
                
                display_name = name.replace("Raw - ", "").replace("Cleaned - ", "")
                
                html_status.append(f"""
                <div class="status-row">
                    <div class="status-left">
                        <span class="status-name">{display_name}</span>
                        <span class="status-meta">{meta_info}</span>
                    </div>
                    <span class="status-pill {css_class}">{icon} {label}</span>
                </div>
                """)
            
            st.markdown("".join(html_status), unsafe_allow_html=True)

    # --- CARD 2: PREFERENCES ---
    with col2:
        with st.container(border=True):
            st.subheader("Preferences")
            st.caption("Global dashboard settings")
            st.write("---")
            
            curr_opts = ["Dollar ($)", "Pound (£)", "Euro (€)", "Rupee (₹)", "Yen (¥)", "None"]
            current_curr = settings.get("currency_label", "Dollar ($)")
            try: def_idx = curr_opts.index(current_curr)
            except: def_idx = 0
                
            new_curr = st.selectbox("Currency Unit", curr_opts, index=def_idx)
            new_unknown = st.toggle("Include 'Unknown' Data", value=settings.get("include_unknown", True))
            
            # Save Logic
            if new_curr != settings.get("currency_label", "") or new_unknown != settings.get("include_unknown", True):
                sym_map = {"Dollar ($)": "$", "Pound (£)": "£", "Euro (€)": "€", "Rupee (₹)": "₹", "Yen (¥)": "¥", "None": ""}
                data_engine.save_setting("currency", sym_map.get(new_curr, ""))
                data_engine.save_setting("currency_label", new_curr)
                data_engine.save_setting("include_unknown", new_unknown)
                st.toast("Settings updated. Refreshing...", icon="⚙️")
                st.rerun()

    # --- CARD 3: PROCESSING ---
    with col3:
        with st.container(border=True):
            st.subheader("Processing")
            st.caption("ETL Pipeline Operations")
            st.write("---")
            
            st.info("Run this when source files change to regenerate the master dataset.")
            
            if st.button("▶ Run ETL Pipeline", type="primary", use_container_width=True):
                with st.spinner("Executing Data Engine..."):
                    df = data_engine.process_and_merge()
                    if df is not None:
                        new_settings = data_engine.load_settings()
                        st.success(f"Build Complete!")
                        st.rerun() 
                    else:
                        st.error("Build Failed. Check logs.")

    # --- BOTTOM: LOG VIEWER ---
    st.write("###")
    with st.expander("System Logs (Console Output)", expanded=False):
        logs = data_engine.get_cleaning_log()
        log_html = []
        for line in logs:
            color = "#ccc"
            if "✅" in line: color = "#4caf50"
            elif "❌" in line: color = "#f44336"
            elif "Warning" in line: color = "#ff9800"
            log_html.append(f"<div style='font-family:monospace; color:{color}; font-size:0.85em; border-bottom:1px solid #222; padding:2px;'>{line}</div>")
        
        st.markdown("".join(log_html), unsafe_allow_html=True)

# ---------------------------------------------------------
# MODULE ROUTING
# ---------------------------------------------------------
elif selected == "Risk Landscape":
    st.title("Risk Landscape") 
    df = data_engine.load_data()
    if df is not None: landscape.show(df)
    else: st.warning("⚠️ Dataset not found. Please go to **Data Sources**.")

elif selected == "Risk Drivers":
    st.title("Risk Drivers")
    df = data_engine.load_data()
    if df is not None: drivers.show(df)
    else: st.warning("⚠️ Dataset not found. Please go to **Data Sources**.")

elif selected == "Deep Dive":
    df = data_engine.load_data()
    if df is not None: deep_dive.show(df)
    else: st.warning("⚠️ Dataset not found. Please go to **Data Sources**.")

elif selected == "Risk Assessment":
    st.title("Credit Assessment")
    df = data_engine.load_data()
    if df is not None: assessment.show(df)
    else: st.warning("⚠️ Dataset not found. Please go to **Data Sources**.")