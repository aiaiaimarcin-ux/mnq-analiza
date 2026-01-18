import streamlit as st
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, time, timedelta, date
import pytz
import os
import math

try:
    from analysis_ib_double_breakout import analyze_ib_double_breakout, calculate_streaks, run_simulation
except ImportError:
    st.error("Błąd: Nie znaleziono pliku 'analysis_ib_double_breakout.py'.")
    st.stop()

# --- KONFIGURACJA WYKRESÓW ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 7  # Mniejsza czcionka ogólna
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['axes.labelcolor'] = 'gray'
plt.rcParams['text.color'] = 'white'
plt.rcParams['xtick.color'] = 'gray'
plt.rcParams['ytick.color'] = 'gray'

st.set_page_config(layout="wide", page_title="MNQ IB Advanced")
st.title("🎯 MNQ – IB Multi-Strategy (Advanced Analytics)")

if 'res' not in st.session_state: st.session_state['res'] = None
if 'df_all' not in st.session_state: st.session_state['df_all'] = None
if 'date_idx' not in st.session_state: st.session_state['date_idx'] = 0
if 'sim_idx' not in st.session_state: st.session_state['sim_idx'] = 0
if 'sim_res' not in st.session_state: st.session_state['sim_res'] = None

# --- DANE ---
DATA_FILENAME = "data.parquet"
if not os.path.exists(DATA_FILENAME):
    for root, dirs, files in os.walk("."):
        if "data.parquet" in files:
            DATA_FILENAME = os.path.join(root, "data.parquet")
            break

def _local_to_utc(date_obj, time_obj):
    tz = pytz.timezone("America/New_York")
    dt_naive = datetime.combine(date_obj, time_obj)
    try: return tz.localize(dt_naive, is_dst=None).astimezone(pytz.UTC)
    except: return tz.localize(dt_naive, is_dst=False).astimezone(pytz.UTC)

@st.cache_resource
def load_data(filepath):
    if not os.path.exists(filepath): return None
    try: return pl.read_parquet(filepath)
    except Exception as e: return str(e)

df_raw = load_data(DATA_FILENAME)

if df_raw is None or isinstance(df_raw, str):
    st.warning(f"⚠️ Nie znaleziono pliku data.parquet")
    uploaded_file = st.sidebar.file_uploader("📂 Wgraj plik ręcznie", type=['parquet'])
    if uploaded_file is not None:
        try:
            df_raw = pl.read_parquet(uploaded_file)
            st.sidebar.success("✅ Wczytano!")
        except: pass

if df_raw is not None and isinstance(df_raw, pl.DataFrame):
    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Ustawienia Danych")
        try:
            ts_col = df_raw.columns[0]
            min_ts = df_raw.select(pl.col(ts_col).min()).item()
            max_ts = df_raw.select(pl.col(ts_col).max()).item()
            if isinstance(min_ts, str):
                 d_start_def = datetime.strptime(min_ts[:8], "%Y%m%d").date()
                 d_end_def = datetime.strptime(max_ts[:8], "%Y%m%d").date()
            else:
                 d_start_def = min_ts.date()
                 d_end_def = max_ts.date()
        except:
            d_start_def = date(2024, 1, 1)
            d_end_def = date.today()

        col_d1, col_d2 = st.columns(2)
        start_d, end_d = col_d1.date_input("Od", d_start_def), col_d2.date_input("Do", d_end_def)
        ib_s, ib_e = st.time_input("Start IB", time(1, 0)), st.time_input("Koniec IB", time(2, 0))
        dead = st.time_input("Deadline", time(17, 0))
        is_ov = st.checkbox("Overnight", value=(ib_s > ib_e))
        b_dir = st.radio("Kierunek Wybicia", ["UP", "DOWN", "BOTH"], index=2)
        b_typ = st.radio("Typ Wybicia", ["wick", "close"], index=0)

        run_btn = st.button("🚀 Uruchom Analizę", type="primary")

    if run_btn:
        with st.spinner("Przetwarzanie danych..."):
            res, df_all = analyze_ib_double_breakout(df_raw, ib_s, ib_e, dead, b_dir, b_typ, is_ov, start_d, end_d)
            st.session_state['res'], st.session_state['df_all'] = res, df_all
            st.session_state['date_idx'] = 0
            st.session_state['sim_idx'] = 0

    if st.session_state['res'] is not None and not st.session_state['res'].is_empty():
        res = st.session_state['res']
        df_all = st.session_state['df_all']
        
        tab1, tab2 = st.tabs(["📊 Statystyki Wybicia", "🎲 Symulator & Wizualizacja"])

        # --- TAB 1: Statystyki ---
        with tab1:
            st.subheader("Wykres i Statystyki Dystrybucji")
            available_dates = sorted(res["date"].unique(), reverse=True)
            c_nav1, c_nav2, c_nav3 = st.columns([1, 2, 1])
            with c_nav1:
                if st.button("⬅️ Poprzedni") and st.session_state['date_idx'] < len(available_dates) - 1:
                    st.session_state['date_idx'] += 1
            with c_nav3:
                if st.button("Następny ➡️") and st.session_state['date_idx'] > 0:
                    st.session_state['date_idx'] -= 1
            with c_nav2:
                current_date = available_dates[st.session_state['date_idx']]
                st.markdown(f"<h4 style='text-align: center;'>{current_date}</h4>", unsafe_allow_html=True)
            row = res.filter(pl.col("date") == current_date).row(0, named=True)
            ny_tz = pytz.timezone("America/New_York")
            pdf = df_all.filter((pl.col("ts_utc") >= row["ib_start_utc"] - timedelta(minutes=60)) & 
                              (pl.col("ts_utc") <= _local_to_utc(current_date, dead) + timedelta(minutes=30))).to_pandas()
            if not pdf.empty:
                plt.style.use('dark_background')
                fig_p, ax_p = plt.subplots(figsize=(10, 3.5), dpi=100)
                fig_p.patch.set_facecolor("#0e1117")
                ax_p.plot(pdf["ts_ny"].dt.to_pydatetime(), pdf["close"], color="white", lw=0.8, alpha=0.9)
                
                # IB High/Low - cienkie szare
                ax_p.axhline(row["ib_high"], color="#B0BEC5", ls="-", lw=0.7, alpha=0.6, label="IB High")
                ax_p.axhline(row["ib_low"], color="#B0BEC5", ls="-", lw=0.7, alpha=0.6, label="IB Low")
                
                ib_s_ny = row["ib_start_utc"].astimezone(ny_tz)
                ib_e_ny = row["ib_end_utc"].astimezone(ny_tz)
                ax_p.axvspan(ib_s_ny, ib_e_ny, color='#4fc3f7', alpha=0.15)
                
                rng = row["ib_range"]
                for m in [1, 2]:
                    ax_p.axhline(row["ib_high"] + rng*m, color="gray", ls=":", lw=0.5, alpha=0.3)
                    ax_p.axhline(row["ib_low"] - rng*m, color="gray", ls=":", lw=0.5, alpha=0.3)
                
                ax_p.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=ny_tz))
                st.pyplot(fig_p)
            st.divider()
            m1, m2, m3 = st.columns(3)
            strat_info = [("dbl", "Double Breakout"), ("mid", "50% Retr."), ("line", "Return Line")]
            for i, (k, label) in enumerate(strat_info):
                with [m1, m2, m3][i]:
                    hits = res.filter(pl.col(f"ret_{k}"))
                    st.metric(label, f"{len(hits)/len(res)*100:.1f}%")

        # --- TAB 2: Symulator ---
        with tab2:
            st.subheader("🛠️ Konfiguracja Strategii")
            
            sc1, sc2 = st.columns([1, 1])
            with sc1:
                st.markdown("##### 1. Parametry Wejścia")
                strat_mode = st.radio("Styl Gry:", ["Kontynuacja (Trend)", "Odwrócenie (Fade)"], index=1, horizontal=True)
                is_fade = (strat_mode == "Odwrócenie (Fade)")
                c1, c2 = st.columns(2)
                trigger_pct = c1.number_input("Trigger (%)", value=30, step=5)
                entry_pct = c2.number_input("Entry (%)", value=25, step=5)
                c3, c4 = st.columns(2)
                def_tp = -50 if is_fade else 100
                tp_pct = c3.number_input("TP (%)", value=def_tp, step=10)
                sl_dist_pct = c4.number_input("SL (Risk %)", value=25, step=5)
                reward = abs(tp_pct - entry_pct)
                risk = sl_dist_pct
                if risk > 0:
                    rr = reward / risk
                    st.caption(f"Estimated R:R = 1:{rr:.2f}")

            with sc2:
                st.markdown("##### 2. Zarządzanie Kapitałem")
                risk_model = st.radio("Model Ryzyka", ["Stała Kwota ($)", "Procent Kapitału (%)"], horizontal=True)
                rc1, rc2 = st.columns(2)
                if risk_model == "Stała Kwota ($)":
                    risk_val = rc1.number_input("Ryzyko na trade ($)", value=100.0, step=10.0)
                    start_cap = rc2.number_input("Kapitał Początkowy ($)", value=10000.0, step=1000.0)
                    sim_risk_type = "FIXED"
                else:
                    risk_val = rc1.number_input("Ryzyko (%)", value=1.0, step=0.1)
                    start_cap = rc2.number_input("Kapitał Początkowy ($)", value=10000.0, step=1000.0)
                    sim_risk_type = "FIXED"
                    risk_val = start_cap * (risk_val / 100.0)
                    st.caption(f"Stałe ryzyko wyliczone z kapitału startowego: ${risk_val:.2f}")

            st.divider()
            
            if st.button("🎲 Symuluj", type="primary"):
                mode_code = "FADE" if is_fade else "TREND"
                sim_res = run_simulation(df_all, res, trigger_pct, entry_pct, tp_pct, sl_dist_pct, dead, 
                                         risk_model=sim_risk_type, risk_value=risk_val, strategy_mode=mode_code)
                st.session_state['sim_res'] = sim_res
                st.session_state['sim_idx'] = 0

            if st.session_state['sim_res'] is not None and not st.session_state['sim_res'].is_empty():
                sim_res = st.session_state['sim_res']
                
                valid_trades = sim_res.filter(pl.col("result").is_in(["WIN", "LOSS", "CLOSE"]))
                invalid_trades = sim_res.filter(pl.col("result").str.contains("INVALID"))
                
                if valid_trades.is_empty():
                    st.warning("Brak zrealizowanych transakcji.")
                    if not invalid_trades.is_empty():
                        st.info(f"Odrzucono {len(invalid_trades)} setupów z powodu Double Breakout.")
                else:
                    # STATYSTYKI
                    total_days = len(res["date"].unique())
                    total_trades = len(valid_trades)
                    wins = valid_trades.filter(pl.col("result") == "WIN")
                    losses = valid_trades.filter(pl.col("result") == "LOSS")
                    num_wins = len(wins)
                    num_losses = len(losses)
                    total_pnl = valid_trades["pnl"].sum()
                    total_r = valid_trades["r_result"].sum()
                    win_rate = num_wins / total_trades * 100
                    
                    # Krzywa Kapitału
                    curve_df = valid_trades.select(["date", "pnl"]).sort("date")
                    equity_curve = [start_cap]
                    dates = [curve_df["date"][0] - timedelta(days=1)]
                    
                    curr_eq = start_cap
                    max_eq = start_cap
                    max_dd_val = 0.0
                    
                    for row in curve_df.iter_rows(named=True):
                        curr_eq += row["pnl"]
                        equity_curve.append(curr_eq)
                        dates.append(row["date"])
                        if curr_eq > max_eq: max_eq = curr_eq
                        dd = curr_eq - max_eq
                        if dd < max_dd_val: max_dd_val = dd
                    
                    max_dd_pct = (max_dd_val / max_eq * 100) if max_eq > 0 else 0.0

                    valid_trades_streaks = valid_trades.with_columns((pl.col("pnl") > 0).alias("is_win"))
                    win_streak, loss_streak = calculate_streaks(valid_trades_streaks, "is_win")

                    st.markdown("### 📊 Wyniki Symulacji")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Net Profit", f"${total_pnl:,.2f}", delta=f"{total_r:.2f} R")
                    m2.metric("Win Rate", f"{win_rate:.1f}%")
                    m3.metric("Profit Factor", f"{wins['pnl'].sum() / abs(losses['pnl'].sum()):.2f}" if not losses.is_empty() else "∞")
                    m4.metric("Max Drawdown", f"${max_dd_val:,.2f}", delta=f"{max_dd_pct:.2f}%", delta_color="inverse")
                    
                    m5, m6, m7, m8 = st.columns(4)
                    m5.metric("Total Trades / Days", f"{total_trades} / {total_days}")
                    m6.metric("Wins / Losses", f"{num_wins} / {num_losses}")
                    m7.metric("Max Win Streak", f"{win_streak}")
                    m8.metric("Max Loss Streak", f"{loss_streak}", delta_color="inverse")

                    st.markdown("##### Krzywa Kapitału")
                    chart_data = pl.DataFrame({"Date": dates[1:], "Equity": equity_curve[1:]})
                    st.line_chart(chart_data.to_pandas(), x="Date", y="Equity", height=300)

                    st.divider()

                    # --- WIZUALIZACJA SZCZEGÓŁOWA ---
                    st.subheader("🔍 Przegląd Transakcji (Wizualizacja)")
                    all_logs = sim_res.sort("date", descending=True)
                    trade_dates = all_logs["date"].to_list()
                    
                    if trade_dates:
                        cn1, cn2, cn3 = st.columns([1, 2, 1])
                        with cn1:
                            if st.button("⬅️ Poprzednia") and st.session_state['sim_idx'] < len(trade_dates) - 1:
                                st.session_state['sim_idx'] += 1
                        with cn3:
                            if st.button("Następna ➡️") and st.session_state['sim_idx'] > 0:
                                st.session_state['sim_idx'] -= 1
                        
                        curr_trade_date = trade_dates[st.session_state['sim_idx']]
                        curr_trade_info = all_logs.filter(pl.col("date") == curr_trade_date).row(0, named=True)
                        res_txt = curr_trade_info["result"]
                        res_color = "green" if res_txt == "WIN" else ("red" if res_txt == "LOSS" else "gray")
                        
                        with cn2:
                            st.markdown(f"<h4 style='text-align: center;'>{curr_trade_date} <span style='color:{res_color}'>[{res_txt}]</span></h4>", unsafe_allow_html=True)
                            if curr_trade_info["pnl"] != 0:
                                st.markdown(f"<div style='text-align: center;'>PnL: <b>${curr_trade_info['pnl']:.2f}</b> ({curr_trade_info['r_result']:.2f} R)</div>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<div style='text-align: center;'><i>{curr_trade_info.get('comment', '')}</i></div>", unsafe_allow_html=True)

                        row_stats = res.filter(pl.col("date") == curr_trade_date).row(0, named=True)
                        ny_tz = pytz.timezone("America/New_York")
                        pdf_sim = df_all.filter((pl.col("ts_utc") >= row_stats["ib_start_utc"] - timedelta(minutes=30)) & 
                                          (pl.col("ts_utc") <= _local_to_utc(curr_trade_date, dead) + timedelta(minutes=30))).to_pandas()
                        
                        if not pdf_sim.empty:
                            fig_s, ax_s = plt.subplots(figsize=(10, 4), dpi=100)
                            fig_s.patch.set_facecolor("#0e1117")
                            
                            # Wykres Ceny - Cienki
                            ax_s.plot(pdf_sim["ts_ny"].dt.to_pydatetime(), pdf_sim["close"], color="white", lw=0.8, alpha=0.9)
                            
                            ib_h, ib_l, ib_rng = row_stats["ib_high"], row_stats["ib_low"], row_stats["ib_range"]
                            direction = row_stats["direction"]

                            # IB LEVELS - Cienkie, Szare, Ciągłe (Tło)
                            ax_s.axhline(ib_h, color="#B0BEC5", ls="-", lw=0.6, alpha=0.5, label="IB High")
                            ax_s.axhline(ib_l, color="#B0BEC5", ls="-", lw=0.6, alpha=0.5, label="IB Low")

                            # Obliczenia poziomów strategii
                            if direction == "UP":
                                base = ib_h
                                trig_lvl = base + (ib_rng * (trigger_pct / 100))
                                ent_lvl  = base + (ib_rng * (entry_pct / 100))
                                if is_fade:
                                    tp_lvl = base + (ib_rng * (tp_pct / 100))
                                    sl_lvl = ent_lvl + (ib_rng * (sl_dist_pct / 100))
                                else:
                                    tp_lvl = base + (ib_rng * (tp_pct / 100))
                                    sl_lvl = ent_lvl - (ib_rng * (sl_dist_pct / 100))
                            else: 
                                base = ib_l
                                trig_lvl = base - (ib_rng * (trigger_pct / 100))
                                ent_lvl  = base - (ib_rng * (entry_pct / 100))
                                if is_fade:
                                    tp_lvl = base - (ib_rng * (tp_pct / 100))
                                    sl_lvl = ent_lvl - (ib_rng * (sl_dist_pct / 100))
                                else:
                                    tp_lvl = base - (ib_rng * (tp_pct / 100))
                                    sl_lvl = ent_lvl + (ib_rng * (sl_dist_pct / 100))

                            # TRIGGER - Fiolet, kropki
                            ax_s.axhline(trig_lvl, color="#E040FB", ls=":", lw=1.0, label="Trigger")
                            
                            if "INVALID" not in res_txt:
                                # ENTRY - Niebieski, Ciągły, Grubszy
                                ax_s.axhline(ent_lvl, color="#2979FF", ls="-", lw=1.2, label="Entry")
                                # TP - Zielony, przerywany
                                ax_s.axhline(tp_lvl, color="#00E676", ls="--", lw=1.0, label="TP")
                                # SL - Czerwony, przerywany
                                ax_s.axhline(sl_lvl, color="#FF1744", ls="--", lw=1.0, label="SL")

                            ib_s_ny = row_stats["ib_start_utc"].astimezone(ny_tz)
                            ib_e_ny = row_stats["ib_end_utc"].astimezone(ny_tz)
                            ax_s.axvspan(ib_s_ny, ib_e_ny, color='#4fc3f7', alpha=0.10)
                            
                            # Legenda na zewnątrz lub w rogu
                            ax_s.legend(fontsize=7, loc="upper right", facecolor='#0e1117', labelcolor='white', framealpha=0.6)
                            ax_s.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=ny_tz))
                            st.pyplot(fig_s)

                    st.divider()

                    st.markdown("### 📋 Dziennik Transakcji")
                    trade_log = sim_res.select([
                        pl.col("date").alias("Data"),
                        pl.col("result").alias("Wynik"),
                        pl.col("pnl").alias("PnL ($)"),
                        pl.col("r_result").alias("R"),
                        pl.col("comment").alias("Komentarz")
                    ]).sort("Data", descending=True)
                    
                    st.dataframe(
                        trade_log.to_pandas(), 
                        use_container_width=True,
                        column_config={
                            "Data": st.column_config.DateColumn("Data", format="YYYY-MM-DD"),
                            "PnL ($)": st.column_config.NumberColumn("PnL ($)", format="$%.2f"),
                            "R": st.column_config.NumberColumn("Wynik R", format="%.2f R"),
                        }
                    )
