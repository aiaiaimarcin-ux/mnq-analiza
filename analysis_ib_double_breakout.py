import polars as pl
from datetime import datetime, time, timedelta, date
import pytz

# --- KONFIGURACJA STREF CZASOWYCH ---
NY_TZ = "America/New_York"
UTC_TZ = "UTC"

def _to_utc_internal(date_obj, time_obj):
    tz = pytz.timezone(NY_TZ)
    dt_naive = datetime.combine(date_obj, time_obj)
    try: 
        return tz.localize(dt_naive, is_dst=None).astimezone(pytz.UTC)
    except: 
        return tz.localize(dt_naive, is_dst=False).astimezone(pytz.UTC)

def _prepare_dataframe(df: pl.DataFrame, ib_start: time, is_overnight: bool) -> pl.DataFrame:
    if df.width == 1:
        col_name = df.columns[0]
        df = df.with_columns(pl.col(col_name).str.split(";").alias("parts")).select([
            pl.col("parts").list.get(0).alias("timestamp"),
            pl.col("parts").list.get(1).cast(pl.Float64).alias("open"),
            pl.col("parts").list.get(2).cast(pl.Float64).alias("high"),
            pl.col("parts").list.get(3).cast(pl.Float64).alias("low"),
            pl.col("parts").list.get(4).cast(pl.Float64).alias("close"),
        ])
    else:
        cols = df.columns
        df = df.select([
            pl.col(cols[0]).alias("timestamp"),
            pl.col(cols[1]).cast(pl.Float64).alias("open"),
            pl.col(cols[2]).cast(pl.Float64).alias("high"),
            pl.col(cols[3]).cast(pl.Float64).alias("low"),
            pl.col(cols[4]).cast(pl.Float64).alias("close"),
        ])

    if df.schema["timestamp"] == pl.String:
        df = df.with_columns(pl.col("timestamp").str.strptime(pl.Datetime, "%Y%m%d %H%M%S", strict=False).dt.replace_time_zone(UTC_TZ).alias("ts_utc"))
    elif df.schema["timestamp"] == pl.Int64: 
         df = df.with_columns(pl.col("timestamp").cast(pl.String).str.strptime(pl.Datetime, "%Y%m%d %H%M%S", strict=False).dt.replace_time_zone(UTC_TZ).alias("ts_utc"))
    else:
        df = df.with_columns(pl.col("timestamp").dt.replace_time_zone(UTC_TZ).alias("ts_utc"))

    df = df.with_columns(pl.col("ts_utc").dt.convert_time_zone(NY_TZ).alias("ts_ny"))
    df = df.with_columns(pl.col("ts_ny").dt.date().alias("calendar_date"))
    
    if is_overnight:
        df = df.with_columns(pl.when(pl.col("ts_ny").dt.time() >= ib_start).then(pl.col("calendar_date") + timedelta(days=1)).otherwise(pl.col("calendar_date")).alias("date"))
    else:
        df = df.with_columns(pl.col("calendar_date").alias("date"))
        
    return df.sort("ts_utc")

def _check_target_advanced(trade_window, direction, target_price, ib_h, ib_l, ib_range, end_ib_utc):
    if trade_window.is_empty(): return False, 0.0, 0.0, None
    condition = pl.col("low") <= target_price if direction == "UP" else pl.col("high") >= target_price
    ret_rows = trade_window.filter(condition).sort("ts_utc")
    max_dist, time_to_ret, returned = 0.0, None, False
    if not ret_rows.is_empty():
        returned = True
        ret_ts = ret_rows.row(0, named=True)["ts_utc"]
        scope_df = trade_window.filter(pl.col("ts_utc") <= ret_ts)
        time_to_ret = int((ret_ts - end_ib_utc).total_seconds() / 60)
    else: scope_df = trade_window
    if direction == "UP":
        local_max = scope_df["high"].max()
        max_dist = local_max - ib_h if local_max else 0.0
    else:
        local_min = scope_df["low"].min()
        max_dist = ib_l - local_min if local_min else 0.0
    dist_pts = max(0.0, float(max_dist))
    dist_pct = (dist_pts / ib_range * 100) if ib_range > 0 else 0.0
    return returned, dist_pts, dist_pct, time_to_ret

def calculate_streaks(res_df, col_name):
    """
    Oblicza maksymalną serię True (Win) i False (Loss) w kolumnie boolean.
    """
    if res_df.is_empty(): return 0, 0
    series = res_df[col_name].to_list()
    max_w = curr_w = max_l = curr_l = 0
    for v in series:
        if v: 
            curr_w += 1
            curr_l = 0
        else: 
            curr_l += 1
            curr_w = 0
        max_w = max(max_w, curr_w)
        max_l = max(max_l, curr_l)
    return max_w, max_l

def _analyze_single_day(day_df, date_val, ib_start, ib_end, return_deadline, breakout_direction, breakout_type, is_overnight):
    start_utc = _to_utc_internal(date_val - timedelta(days=1) if is_overnight else date_val, ib_start)
    end_utc = _to_utc_internal(date_val, ib_end)
    deadline_utc = _to_utc_internal(date_val, return_deadline)
    ib_df = day_df.filter((pl.col("ts_utc") >= start_utc) & (pl.col("ts_utc") < end_utc))
    after_df = day_df.filter((pl.col("ts_utc") >= end_utc) & (pl.col("ts_utc") <= deadline_utc))
    if ib_df.is_empty() or after_df.is_empty(): return None
    ib_h, ib_l = ib_df["high"].max(), ib_df["low"].min()
    ib_range = ib_h - ib_l
    ib_mid = ib_l + (ib_range / 2)
    col_up, col_dw = ("high", "low") if breakout_type == "wick" else ("close", "close")
    breaks_df = after_df.filter((pl.col(col_up) > ib_h) | (pl.col(col_dw) < ib_l)).sort("ts_utc")
    if breaks_df.is_empty(): return None
    first_break = breaks_df.row(0, named=True)
    f_dir = "UP" if first_break[col_up] > ib_h else "DOWN"
    if breakout_direction != "BOTH" and f_dir != breakout_direction: return None
    brk_ts = first_break["ts_utc"]
    trade_window = after_df.filter(pl.col("ts_utc") > brk_ts)
    targets = {"dbl": ib_l if f_dir == "UP" else ib_h, "mid": ib_mid, "line": ib_h if f_dir == "UP" else ib_l}
    
    res_dict = {
        "date": date_val, "direction": f_dir, "ib_range": ib_range, 
        "ib_high": ib_h, "ib_low": ib_l, "ib_mid": ib_mid, 
        "ib_start_utc": start_utc, "ib_end_utc": end_utc
    }
    
    for key, price in targets.items():
        hit, d_pts, d_pct, t_min = _check_target_advanced(trade_window, f_dir, price, ib_h, ib_l, ib_range, end_utc)
        res_dict.update({f"ret_{key}": hit, f"dist_{key}": d_pts, f"dist_pct_{key}": d_pct, f"time_{key}": t_min})
    return res_dict

def analyze_ib_double_breakout(df, ib_start, ib_end, return_deadline, breakout_direction="BOTH", breakout_type="wick", is_overnight=False, start_date=None, end_date=None):
    df = _prepare_dataframe(df, ib_start, is_overnight)
    results = []
    for key, day_df in df.group_by("date", maintain_order=True):
        date_val = key[0] if isinstance(key, (tuple, list)) else key
        if date_val is None or day_df.height < 10: continue
        if (start_date and date_val < start_date) or (end_date and date_val > end_date): continue
        res = _analyze_single_day(day_df, date_val, ib_start, ib_end, return_deadline, breakout_direction, breakout_type, is_overnight)
        if res: results.append(res)
    return (pl.DataFrame(results) if results else pl.DataFrame()), df

def run_simulation(df_all, res_df, trigger_pct, entry_pct, tp_pct, sl_dist_pct, deadline, risk_model="FIXED", risk_value=100.0, strategy_mode="TREND"):
    trades = []
    
    for row in res_df.to_dicts():
        date_val = row["date"]
        direction = row["direction"]
        ib_h, ib_l = row["ib_high"], row["ib_low"]
        rng = row["ib_range"]
        
        if direction == "UP":
            invalidation_price = ib_l 
            base = ib_h
            trigger_price = base + (rng * (trigger_pct / 100))
            entry_price   = base + (rng * (entry_pct / 100))
            if strategy_mode == "TREND":
                tp_price = base + (rng * (tp_pct / 100))
                sl_price = entry_price - (rng * (sl_dist_pct / 100))
                trade_dir = "LONG"
            else: 
                tp_price = base + (rng * (tp_pct / 100))
                sl_price = entry_price + (rng * (sl_dist_pct / 100))
                trade_dir = "SHORT"
        else: 
            invalidation_price = ib_h
            base = ib_l
            trigger_price = base - (rng * (trigger_pct / 100))
            entry_price   = base - (rng * (entry_pct / 100))
            if strategy_mode == "TREND":
                tp_price = base - (rng * (tp_pct / 100))
                sl_price = entry_price + (rng * (sl_dist_pct / 100))
                trade_dir = "SHORT"
            else: 
                tp_price = base - (rng * (tp_pct / 100))
                sl_price = entry_price - (rng * (sl_dist_pct / 100))
                trade_dir = "LONG"

        risk_dist = abs(entry_price - sl_price)
        if risk_dist == 0: risk_dist = 1.0
        risk_cash = risk_value if risk_model == "FIXED" else 100.0

        day_df = df_all.filter(pl.col("date") == date_val)
        deadline_utc = _to_utc_internal(date_val, deadline)
        trading_data = day_df.filter((pl.col("ts_utc") >= row["ib_end_utc"]) & (pl.col("ts_utc") <= deadline_utc))
        
        if trading_data.is_empty(): continue

        if direction == "UP":
            trigger_hit = trading_data.filter(pl.col("high") >= trigger_price)
        else:
            trigger_hit = trading_data.filter(pl.col("low") <= trigger_price)

        if trigger_hit.is_empty():
            if direction == "UP":
                dbl_break = trading_data.filter(pl.col("low") < invalidation_price)
            else:
                dbl_break = trading_data.filter(pl.col("high") > invalidation_price)
            res_str = "INVALID (Double Break)" if not dbl_break.is_empty() else "NO TRIGGER"
            trades.append({"date": date_val, "result": res_str, "pnl": 0.0, "r_result": 0.0, "comment": "Trigger missed"})
            continue
            
        trigger_ts = trigger_hit["ts_utc"][0]
        
        pre_trigger_data = trading_data.filter(pl.col("ts_utc") < trigger_ts)
        is_invalid = False
        if not pre_trigger_data.is_empty():
            if direction == "UP":
                if not pre_trigger_data.filter(pl.col("low") < invalidation_price).is_empty(): is_invalid = True
            else:
                if not pre_trigger_data.filter(pl.col("high") > invalidation_price).is_empty(): is_invalid = True
        
        if is_invalid:
            trades.append({"date": date_val, "result": "INVALID (Double Break)", "pnl": 0.0, "r_result": 0.0, "comment": "Setup cancelled"})
            continue

        after_trigger = trading_data.filter(pl.col("ts_utc") > trigger_ts)
        if after_trigger.is_empty():
            trades.append({"date": date_val, "result": "NO TIME", "pnl": 0.0, "r_result": 0.0, "comment": "End of day"})
            continue

        if direction == "UP":
            entry_hit = after_trigger.filter(pl.col("low") <= entry_price)
        else:
            entry_hit = after_trigger.filter(pl.col("high") >= entry_price)
            
        if entry_hit.is_empty():
            trades.append({"date": date_val, "result": "MISSED", "pnl": 0.0, "r_result": 0.0, "comment": "Entry missed"})
            continue 
            
        entry_ts = entry_hit["ts_utc"][0]
        in_trade = after_trigger.filter(pl.col("ts_utc") > entry_ts)
        
        if in_trade.is_empty():
            close_price = entry_hit["close"][0]
            pnl = (close_price - entry_price) if trade_dir == "LONG" else (entry_price - close_price)
            calc_r = pnl / risk_dist
            pnl_cash = calc_r * risk_cash
            trades.append({"date": date_val, "result": "CLOSE", "pnl": pnl_cash, "r_result": calc_r, "comment": "EOD"})
            continue

        outcome = "CLOSE"
        exit_price = in_trade["close"][-1]
        
        for candle in in_trade.iter_rows(named=True):
            if trade_dir == "LONG":
                hit_sl = candle["low"] <= sl_price
                hit_tp = candle["high"] >= tp_price
            else:
                hit_sl = candle["high"] >= sl_price
                hit_tp = candle["low"] <= tp_price
            
            if hit_sl and hit_tp:
                outcome = "LOSS"
                exit_price = sl_price
                break
            elif hit_sl:
                outcome = "LOSS"
                exit_price = sl_price
                break
            elif hit_tp:
                outcome = "WIN"
                exit_price = tp_price
                break
        
        pnl_pts = (exit_price - entry_price) if trade_dir == "LONG" else (entry_price - exit_price)
        r_res = pnl_pts / risk_dist
        pnl_cash = r_res * risk_cash
        trades.append({"date": date_val, "result": outcome, "pnl": pnl_cash, "r_result": r_res, "comment": f"{outcome}"})
        
    return pl.DataFrame(trades)
