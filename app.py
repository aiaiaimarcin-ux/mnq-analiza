import streamlit as st
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, time, timedelta, date
import pytz
import os
import math

from analysis_ib_double_breakout import analyze_ib_double_breakout, calculate_streaks

st.set_page_config(layout="wide", page_title="MNQ IB Advanced")
st.title("🎯 MNQ – IB Multi-Strategy (Advanced Analytics)")

if 'res' not in st.session_state: st.session_state['res'] = None
if 'df_all' not in st.session_state: st.session_state['df_all'] = None
if 'date_idx' not in st.session_state: st.session_state['date_idx'] = 0

DATA_FILENAME = "data.txt"

def _local_to_utc(date_obj, time_obj):
    tz = pytz.timezone("America/New_York")
    dt_naive = datetime.combine(date_obj, time_obj)
    try: return tz.localize(dt_naive, is_dst=None).astimezone(pytz.UTC)
    except: return tz.localize(dt_naive, is_dst=False).astimezone(pytz.UTC)

def format_minutes(m):
    # Sprawdzenie czy m to None, albo czy jest NaN (Not a Number)
    if m is None: return "-"
    if isinstance(m, float) and math.isnan(m): return "-"
    
    try:
        val = float(m)
        h = int(val // 60)
        mn = int(val % 60)
        return f"{h}h {mn}m"
    except:
        return "-"

@st.cache_resource
def load_data(filepath):
    if not os.path.exists(filepath): return None
    try: return pl.read_csv(filepath, has_header=False)
    except Exception as e: return str(e)

df_raw = load_data(DATA_FILENAME)

if df_raw is None:
    st.error(f"⚠️ Brak pliku {DATA_FILENAME}")
else:
    with st.sidebar:
        st.header("⚙️ Ustawienia")
        col_d1, col_d2 = st.columns(2)
        start_d, end_d = col_d1.date_input("Od", date(2024, 1, 1)), col_d2.date_input("Do", date.today())
        ib_s, ib_e = st.time_input("Start IB", time(1, 0)), st.time_input("Koniec IB", time(2, 0))
        dead = st.time_input("Deadline", time(17, 0))
        is_ov = st.checkbox("Overnight", value=(ib_s > ib_e))
        b_dir = st.radio("Kierunek", ["UP", "DOWN", "BOTH"], index=2)
        b_typ = st.radio("Typ", ["wick", "close"], index=0)

        if st.button("🚀 Uruchom Analizę", type="primary"):
            with st.spinner("Liczenie..."):
                res, df_all = analyze_ib_double_breakout(df_raw, ib_s, ib_e, dead, b_dir, b_typ, is_ov, start_d, end_d)
                st.session_state['res'], st.session_state['df_all'] = res, df_all
                st.session_state['date_idx'] = 0

    if st.session_state['res'] is not None and not st.session_state['res'].is_empty():
        res = st.session_state['res']
        df_all = st.session_state['df_all']

        # STATYSTYKI
        st.subheader("📊 Kluczowe Statystyki")
        m1, m2, m3 = st.columns(3)
        strat_info = [("dbl", "Double Breakout"), ("mid", "50% Retr."), ("line", "Return Line")]
        
        for i, (k, label) in enumerate(strat_info):
            with [m1, m2, m3][i]:
                hits = res.filter(pl.col(f"ret_{k}"))
                ws, ls = calculate_streaks(res, f"ret_{k}")
                st.metric(label, f"Win: {len(hits)/len(res)*100:.1f}%")
                if not hits.is_empty():
                    st.write(f"📏 Średnie: **{hits[f'dist_{k}'].mean():.1f} pkt** ({hits[f'dist_pct_{k}'].mean():.1f}% IB)")
                    st.write(f"🎯 Mediana: **{hits[f'dist_{k}'].median():.1f} pkt** ({hits[f'dist_pct_{k}'].median():.1f}%)")
                    st.write(f"🔥 Streaks: **W:{ws} | L:{ls}**")

        st.divider()

        # NAWIGACJA STRZAŁKAMI NA WYKRESIE
        st.subheader("🔍 Przegląd Sesji")
        available_dates = sorted(res["date"].unique(), reverse=True)
        
        col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
        with col_nav1:
            if st.button("⬅️ Dzień Wcześniej") and st.session_state['date_idx'] < len(available_dates) - 1:
                st.session_state['date_idx'] += 1
        with col_nav3:
            if st.button("Dzień Później ➡️") and st.session_state['date_idx'] > 0:
                st.session_state['date_idx'] -= 1
        with col_nav2:
            current_date = available_dates[st.session_state['date_idx']]
            st.markdown(f"<h3 style='text-align: center;'>{current_date}</h3>", unsafe_allow_html=True)

        row = res.filter(pl.col("date") == current_date).row(0, named=True)
        ny_tz = pytz.timezone("America/New_York")
        
        pdf = df_all.filter((pl.col("ts_utc") >= row["ib_start_utc"] - timedelta(minutes=60)) & 
                          (pl.col("ts_utc") <= _local_to_utc(current_date, dead) + timedelta(minutes=30))).to_pandas()
        
        if not pdf.empty:
            plt.style.use('dark_background')
            fig_p, ax_p = plt.subplots(figsize=(10, 4))
            fig_p.patch.set_facecolor("#0e1117")
            ax_p.plot(pdf["ts_ny"].dt.to_pydatetime(), pdf["close"], color="white", lw=1.2, label="Cena")
            
            # Linie IB
            ax_p.axhline(row["ib_high"], color="#4CAF50", ls="--", alpha=0.7, label="IB High")
            ax_p.axhline(row["ib_low"], color="#FF5252", ls="--", alpha=0.7, label="IB Low")

            # Linie rozszerzeń (Extensions) 1x, 2x, 3x IB
            rng = row["ib_range"]
            for mult in [1, 2, 3]:
                ax_p.axhline(row["ib_high"] + (rng * mult), color="#606060", ls=":", lw=0.8, alpha=0.5)
                ax_p.axhline(row["ib_low"] - (rng * mult), color="#606060", ls=":", lw=0.8, alpha=0.5)
            
            # Jaśniejszy prostokąt IB (#4fc3f7 - Light Blue)
            ib_s_ny = row["ib_start_utc"].astimezone(ny_tz)
            ib_e_ny = row["ib_end_utc"].astimezone(ny_tz)
            ax_p.axvspan(ib_s_ny, ib_e_ny, color='#4fc3f7', alpha=0.25, label="Strefa IB")
            
            ax_p.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=ny_tz))
            ax_p.legend(fontsize=8, loc='upper left')
            st.pyplot(fig_p)

        # WYKRESY KOŁOWE
        st.divider()
        st.subheader("🥧 Skuteczność i Rozkład Czasu")
        sel_strat = st.selectbox("Wybierz strategię do wykresów i tabeli:", ["Double Breakout", "50% Retr.", "Return Line"])
        k_key = {"Double Breakout": "dbl", "50% Retr.": "mid", "Return Line": "line"}[sel_strat]
        
        c_pie1, c_pie2 = st.columns(2)
        with c_pie1:
            win_c = res.filter(pl.col(f"ret_{k_key}")).height
            loss_c = res.height - win_c
            fig1, ax1 = plt.subplots(figsize=(3, 3))
            fig1.patch.set_facecolor("#0e1117")
            ax1.pie([win_c, loss_c], labels=['Sukces', 'Brak'], autopct='%1.1f%%', colors=['#2e7d32', '#c62828'], startangle=90)
            ax1.set_title(f"Skuteczność: {sel_strat}")
            st.pyplot(fig1)
        
        with c_pie2:
            time_hits = res.filter(pl.col(f"ret_{k_key}") & pl.col(f"time_{k_key}").is_not_null())
            if not time_hits.is_empty():
                hours = ((time_hits[f"time_{k_key}"] / 60).floor().cast(pl.Int32) + 1).alias("hours_bucket")
                counts = hours.value_counts().sort("hours_bucket")
                fig2, ax2 = plt.subplots(figsize=(3, 3))
                fig2.patch.set_facecolor("#0e1117")
                ax2.pie(counts["count"], labels=[f"{int(h)}h" for h in counts["hours_bucket"]], autopct='%1.1f%%', startangle=140)
                ax2.set_title(f"Czas powrotu: {sel_strat}")
                st.pyplot(fig2)

        # SZCZEGÓŁOWA TABELA
        st.divider()
        st.subheader(f"📋 Szczegółowa lista transakcji: {sel_strat}")
        
        table_df = res.select([
            pl.col("date").alias("Data"),
            pl.col("direction").alias("Kierunek"),
            pl.col("ib_range").round(2).alias("IB Range"),
            pl.col(f"ret_{k_key}").alias("Wynik"),
            pl.col(f"dist_{k_key}").round(2).alias("Max Odchylenie (pkt)"),
            pl.col(f"dist_pct_{k_key}").round(1).alias("Max Odchylenie (%IB)"),
            pl.col(f"time_{k_key}").alias("mins_raw")
        ]).sort("Data", descending=True)
        
        pdf_table = table_df.to_pandas()
        
        pdf_table["Czas powrotu"] = pdf_table["mins_raw"].apply(format_minutes)
        pdf_table["Wynik"] = pdf_table["Wynik"].map({True: "✅ WIN", False: "❌ LOSS"})
        
        pdf_table = pdf_table.drop(columns=["mins_raw"])

        st.dataframe(
            pdf_table, 
            use_container_width=True,
            column_config={
                "Data": st.column_config.DateColumn("Data", format="YYYY-MM-DD"),
                "Max Odchylenie (%IB)": st.column_config.NumberColumn("Odchylenie %", format="%.1f%%"),
            },
            height=400
        )