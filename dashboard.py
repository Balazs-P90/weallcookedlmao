"""
Tactical, comprehensive dashboard for End-of-World Predictor results.
Displays scenario explorer, risk summaries, uncertainty, and feature importances.
"""
import streamlit as st
import pandas as pd
import json
import os
from glob import glob


def load_latest_results(results_dir='results'):
    files = sorted(glob(os.path.join(results_dir, 'mc_results_*.csv')))
    if not files:
        st.warning('No results found.')
        return None
    df = pd.read_csv(files[-1])
    return df

def load_summary(results_dir='results'):
    files = sorted(glob(os.path.join(results_dir, 'eow_predictor.log')))
    if not files:
        return None
    # Parse last summary from log
    with open(files[-1], 'r') as f:
        lines = f.readlines()
    summary_lines = [l for l in lines if 'Summary:' in l]
    if not summary_lines:
        return None
    last_summary = summary_lines[-1].split('Summary:')[-1].strip()
    try:
        return json.loads(last_summary)
    except Exception:
        return None

def main():
    from glob import glob
    def table_svg():
        return '''<svg viewBox="0 0 32 32" width="1.3em" height="1.3em"><rect x="4" y="8" width="24" height="16" fill="#23282d" stroke="#b7ff4d" stroke-width="2"/><line x1="4" y1="16" x2="28" y2="16" stroke="#b7ff4d" stroke-width="2"/><line x1="12" y1="8" x2="12" y2="24" stroke="#b7ff4d" stroke-width="2"/><line x1="20" y1="8" x2="20" y2="24" stroke="#b7ff4d" stroke-width="2"/></svg>'''

    # Load results and summary at the top so they are available everywhere
    df = load_latest_results()
    summary = load_summary()

    st.set_page_config(page_title='EOW Command Dashboard', layout='wide', initial_sidebar_state='expanded')
    # --- SVG ICONS (must be defined before use) ---
    def nuclear_svg():
        return '''<svg viewBox="0 0 32 32" width="1.3em" height="1.3em"><circle cx="16" cy="16" r="15" fill="#23282d" stroke="#b7ff4d" stroke-width="2"/><path d="M16 16 L28 16 A12 12 0 0 0 16 4 Z" fill="#b7ff4d"/><path d="M16 16 L16 4 A12 12 0 0 0 4 16 Z" fill="#b7ff4d"/><path d="M16 16 L4 16 A12 12 0 0 0 16 28 Z" fill="#b7ff4d"/></svg>'''
    def famine_svg():
        return '''<svg viewBox="0 0 32 32" width="1.3em" height="1.3em"><ellipse cx="16" cy="24" rx="10" ry="6" fill="#b7ff4d"/><rect x="14" y="4" width="4" height="20" fill="#b7ff4d"/></svg>'''
    def war_svg():
        return '''<svg viewBox="0 0 32 32" width="1.3em" height="1.3em"><rect x="6" y="14" width="20" height="4" fill="#b7ff4d"/><rect x="14" y="6" width="4" height="20" fill="#b7ff4d"/></svg>'''
    if summary is not None:
        st.markdown('<div class="tac-panel"><span class="tac-icon">🛡️</span><b>Key Risk Metrics</b></div>', unsafe_allow_html=True)
        def risk_color(val, reverse=False):
            if val is None or val != val:
                return '#23282d'
            if reverse:
                if val < 5: return '#ff4d4d'
                if val < 15: return '#ffa94d'
                return '#b7ff4d'
            else:
                if val > 0.5: return '#ff4d4d'
                if val > 0.1: return '#ffa94d'
                return '#b7ff4d'
        def risk_status(val, reverse=False):
            if val is None or val != val:
                if summary is not None:
                    st.markdown('<div class="tac-panel"><span class="tac-icon">🛡️</span><b>Key Risk Metrics</b></div>', unsafe_allow_html=True)

    df = load_latest_results()
    summary = load_summary()

    # --- Scenario Explorer ---
    st.sidebar.markdown('<div class="tac-panel"><span class="tac-icon">' + nuclear_svg() + '</span><b>Scenario Explorer</b></div>', unsafe_allow_html=True)
    scenario_templates = {
        'Default': json.dumps({
            "temp_anomaly": 0.9,
            "temp_trend_decade": 0.2,
            "conflict_score": 2.0,
            "nuclear_deployed": 2000,
            "mobility": 1.0,
            "zoonotic_events": 1,
            "vax_coverage": 0.7,
            "ai_incidents": 0,
            "ai_incident_score": 0.0,
            "shipping_delays": 0,
            "shortages": 0
        }, indent=2),
        'Extreme Climate': '{"temp_anomaly": 2.5, "temp_trend_decade": 0.7}',
        'Major Conflict': '{"conflict_score": 7.0, "nuclear_deployed": 12000}',
        'Pandemic': '{"mobility": 0.8, "zoonotic_events": 5, "vax_coverage": 0.2}',
        'AI Catastrophe': '{"ai_incidents": 10, "ai_incident_score": 0.9}',
        'Supply Chain Collapse': '{"shipping_delays": 10, "shortages": 8}',
    }
    template = st.sidebar.selectbox('Scenario template', list(scenario_templates.keys()), index=0)
    scenario = st.sidebar.text_area('Custom scenario (JSON)', value=scenario_templates[template], height=180)
    import subprocess, time
    if st.sidebar.button('Run Scenario'):
        # Validate scenario JSON
        try:
            scenario_dict = json.loads(scenario)
        except Exception as e:
            st.sidebar.error(f"Invalid scenario JSON: {e}")
            return
        st.sidebar.info('Running scenario simulation...')
        # Write scenario to a temp file to avoid shell escaping issues
        import tempfile
        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.json') as tf:
            json.dump(scenario_dict, tf)
            tf.flush()
            scenario_path = tf.name
        # Run the simulation and wait for it to finish
        cmd = ["python", "end_of_world_predictor.py", "--scenario", f"@{scenario_path}", "--once"]
        # Support @file for scenario input in end_of_world_predictor.py (patch needed if not present)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            st.sidebar.error(f"Simulation failed: {proc.stderr}")
        else:
            st.sidebar.success('Scenario simulation complete!')
            # Wait for new results file to appear (max 10s)
            from glob import glob
            prev_files = set(glob('results/mc_results_*.csv'))
            for _ in range(20):
                time.sleep(0.5)
                new_files = set(glob('results/mc_results_*.csv'))
                if len(new_files - prev_files) > 0:
                    break
            st.rerun()

    # --- Risk Cards ---
    def risk_card(label, value, unit, color, helptext=None, icon=None, status=None):
        status_html = ''
        if status == 'red':
            status_html = '<span class="status-light status-red"></span>'
        elif status == 'yellow':
            status_html = '<span class="status-light status-yellow"></span>'
        elif status == 'green':
            status_html = '<span class="status-light status-green"></span>'
        icon_html = icon or ''
        st.markdown(f"""
        <div class="risk-card">
        {status_html}<span class="tac-icon">{icon_html}</span><span style='font-size:1.3em;font-weight:bold'>{label}:</span>
        <span style='font-size:1.3em;float:right'>{value} {unit}</span><br>
        <span style='font-size:0.95em;color:#b7ff4d'>{helptext or ''}</span>
        </div>
        """, unsafe_allow_html=True)

    # --- SVG ICONS ---
    def nuclear_svg():
        return '''<svg viewBox="0 0 32 32" width="1.3em" height="1.3em"><circle cx="16" cy="16" r="15" fill="#23282d" stroke="#b7ff4d" stroke-width="2"/><path d="M16 16 L28 16 A12 12 0 0 0 16 4 Z" fill="#b7ff4d"/><path d="M16 16 L16 4 A12 12 0 0 0 4 16 Z" fill="#b7ff4d"/><path d="M16 16 L4 16 A12 12 0 0 0 16 28 Z" fill="#b7ff4d"/></svg>'''
    def famine_svg():
        return '''<svg viewBox="0 0 32 32" width="1.3em" height="1.3em"><ellipse cx="16" cy="24" rx="10" ry="6" fill="#b7ff4d"/><rect x="14" y="4" width="4" height="20" fill="#b7ff4d"/></svg>'''
    def war_svg():
        return '''<svg viewBox="0 0 32 32" width="1.3em" height="1.3em"><rect x="6" y="14" width="20" height="4" fill="#b7ff4d"/><rect x="14" y="6" width="4" height="20" fill="#b7ff4d"/></svg>'''
    def alert_svg():
        return '''<svg viewBox="0 0 32 32" width="1.3em" height="1.3em"><circle cx="16" cy="16" r="15" fill="#23282d" stroke="#b7ff4d" stroke-width="2"/><rect x="14" y="8" width="4" height="12" fill="#b7ff4d"/><rect x="14" y="22" width="4" height="4" fill="#b7ff4d"/></svg>'''
    def temp_svg():
        return '''<svg viewBox="0 0 32 32" width="1.3em" height="1.3em"><rect x="14" y="6" width="4" height="16" fill="#b7ff4d"/><ellipse cx="16" cy="24" rx="8" ry="4" fill="#b7ff4d"/></svg>'''

    if summary is not None:
        st.markdown('<div class="tac-panel"><span class="tac-icon">🛡️</span><b>Key Risk Metrics</b></div>', unsafe_allow_html=True)
        def risk_color(val, reverse=False):
            if val is None or val != val:
                return '#23282d'
            if reverse:
                if val < 5: return '#ff4d4d'
                if val < 15: return '#ffa94d'
                return '#b7ff4d'
            else:
                if val > 0.5: return '#ff4d4d'
                if val > 0.1: return '#ffa94d'
                return '#b7ff4d'
        def risk_status(val, reverse=False):
            if val is None or val != val:
                return 'yellow'
            if reverse:
                if val < 5: return 'red'
                if val < 15: return 'yellow'
                return 'green'
            else:
                if val > 0.5: return 'red'
                if val > 0.1: return 'yellow'
                return 'green'
        col1, col2, col3 = st.columns(3)
        with col1:
            risk_card('Nuclear War (10y)', f"{summary.get('prob_nuclear_war_10y',0):.2%}", '', risk_color(summary.get('prob_nuclear_war_10y',0)), 'Probability of nuclear war within 10 years', icon=nuclear_svg(), status=risk_status(summary.get('prob_nuclear_war_10y',0)))
            risk_card('Famine (10y)', f"{summary.get('prob_famine_10y',0):.2%}", '', risk_color(summary.get('prob_famine_10y',0)), 'Probability of global famine within 10 years', icon=famine_svg(), status=risk_status(summary.get('prob_famine_10y',0)))
            # --- Famine Risk Explainability ---
            # Try to load famine explain dict from the latest log file
            log_files = sorted(glob('results/eow_predictor.log'))
            famine_explain = None
            if log_files:
                with open(log_files[-1], 'r') as f:
                    lines = f.readlines()
                # Find the most recent famine_explain output if present
                for l in reversed(lines):
                    if 'famine_explain' in l:
                        try:
                            famine_explain = json.loads(l.split('famine_explain:')[-1].strip())
                            break
                        except Exception:
                            continue
            if famine_explain:
                st.markdown('<div class="tac-panel"><b>Famine Risk Explainability</b></div>', unsafe_allow_html=True)
                st.json(famine_explain)
            else:
                st.markdown('<div class="tac-panel"><b>Famine Risk Explainability</b><br><span style="color:#b7ff4d">No recent explainability data found. Run a scenario or simulation to update.</span></div>', unsafe_allow_html=True)
        with col2:
            risk_card('World War (10y)', f"{summary.get('prob_war_10y',0):.2%}", '', risk_color(summary.get('prob_war_10y',0)), 'Probability of world war within 10 years', icon=war_svg(), status=risk_status(summary.get('prob_war_10y',0)))
            risk_card('Any Major Event (10y)', f"{summary.get('prob_any_major_event_10y',0):.2%}", '', risk_color(summary.get('prob_any_major_event_10y',0)), 'Probability of any major event within 10 years', icon=alert_svg(), status=risk_status(summary.get('prob_any_major_event_10y',0)))
        with col3:
            risk_card('Years to 2°C', f"{summary.get('years_to_2C_1_99',(0,0))[0]:.1f}-{summary.get('years_to_2C_1_99',(0,0))[1]:.1f}", 'yrs', risk_color(summary.get('years_to_2C_1_99',(0,0))[0], reverse=True), '1st–99th percentile years to 2°C warming', icon=temp_svg(), status=risk_status(summary.get('years_to_2C_1_99',(0,0))[0], reverse=True))

        st.markdown('<hr style="border:1px solid #b7ff4d;">', unsafe_allow_html=True)
    st.markdown('<div class="tac-panel"><span class="tac-icon">' + war_svg() + '</span><b>Risk Distribution</b></div>', unsafe_allow_html=True)
    if df is not None:
        # Show only one risk metric for maximum simplicity
        for col in ['nuclear_median_years', 'famine_median_years', 'war_median_years']:
            if col in df.columns:
                st.line_chart(df[col], height=120, use_container_width=True)
                break
        st.markdown('<hr style="border:1px solid #b7ff4d;">', unsafe_allow_html=True)
    st.markdown('<div class="tac-panel"><span class="tac-icon">' + temp_svg() + '</span><b>Summary Statistics</b></div>', unsafe_allow_html=True)
    st.json(summary)
    st.markdown('<hr style="border:1px solid #b7ff4d;">', unsafe_allow_html=True)
    st.markdown('<div class="tac-panel"><span class="tac-icon">' + alert_svg() + '</span><b>Event Frequencies & Warnings</b></div>', unsafe_allow_html=True)
    if 'freq' in summary:
        for k, v in summary['freq'].items():
            st.write(f"**{k}**: {v:.2%}")

    # --- Feature Importances ---
    st.sidebar.markdown('<div class="tac-panel"><span class="tac-icon">' + temp_svg() + '</span><b>Explainability</b></div>', unsafe_allow_html=True)
    if os.path.exists('results/eow_predictor.log'):
        with open('results/eow_predictor.log', 'r') as f:
            lines = f.readlines()
        feat_lines = [l for l in lines if 'feature importances' in l]
        if feat_lines:
            st.sidebar.markdown('**Feature Importances:**')
            for l in feat_lines[-3:]:
                st.sidebar.code(l)

    # --- Data Table ---
    if df is not None:
        st.markdown('<hr style="border:1px solid #b7ff4d;">', unsafe_allow_html=True)
    st.markdown('<div class="tac-panel"><span class="tac-icon">' + table_svg() + '</span><b>Raw Monte Carlo Results (first 100 rows)</b></div>', unsafe_allow_html=True)
    st.dataframe(df.head(100), use_container_width=True)

if __name__ == '__main__':
    main()
