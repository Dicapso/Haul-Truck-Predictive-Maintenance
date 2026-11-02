"""Cleaning, feature engineering and alert evaluation for the haul-truck predictive-maintenance project."""
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
SENS = ["coolant_temp_c", "oil_pressure_kpa", "trans_oil_temp_c", "brake_temp_c",
        "hydraulic_pressure_bar", "vibration_rms_mm_s", "fuel_rate_lph"]
# direction in which each sensor moves when a component degrades
SIGN = {"coolant_temp_c": 1, "oil_pressure_kpa": -1, "trans_oil_temp_c": 1, "brake_temp_c": 1,
        "hydraulic_pressure_bar": -1, "vibration_rms_mm_s": 1, "fuel_rate_lph": 1}
HORIZON = 7               # predict a breakdown in the next 1..7 days
MARGIN_PER_H = 420        # AZN lost haulage margin per truck-hour of unplanned downtime (assumption)
INSPECTION_AZN = 4 * MARGIN_PER_H + 300   # 4 h out of the cycle + technician time
P_FIND = 0.8              # chance an inspection finds a developing fault
PLANNED_COST, PLANNED_DOWN = 0.35, 0.30   # planned repair vs unplanned: share of cost and downtime


def load_raw():
    return (pd.read_csv(DATA / "telemetry_daily.csv"), pd.read_csv(DATA / "maintenance_log.csv"),
            pd.read_csv(DATA / "fleet.csv", parse_dates=["commissioned"]))


def clean(tele):
    df, log = tele.copy(), {"Raw telemetry rows": len(tele)}
    tid = df.truck_id.str.strip().str.upper().str.extract(r"HT-(\d+)")[0].astype(int)
    log["Truck IDs re-formatted (case / spaces / zero-padding)"] = int((df.truck_id != tid.map("HT-{:02d}".format)).sum())
    df["truck_id"] = tid.map("HT-{:02d}".format)
    iso = pd.to_datetime(df.date, format="%Y-%m-%d", errors="coerce")
    dmy = pd.to_datetime(df.date, format="%d.%m.%Y", errors="coerce")
    log["Dates in dd.mm.yyyy format parsed"] = int(iso.isna().sum())
    df["date"] = iso.fillna(dmy)
    n = len(df)
    df = df.drop_duplicates(["truck_id", "date"]).sort_values(["truck_id", "date"]).reset_index(drop=True)
    log["Duplicate truck-day rows dropped"] = n - len(df)
    f = df.coolant_temp_c > 150
    log["Coolant readings in °F converted"] = int(f.sum())
    df.loc[f, "coolant_temp_c"] = ((df.loc[f, "coolant_temp_c"] - 32) * 5 / 9).round(1)
    # stuck sensor: the same value five or more operating days in a row
    stuck = pd.Series(False, index=df.index)
    for c in SENS:
        g = (df[c] != df.groupby("truck_id")[c].shift()).cumsum()
        run = df.groupby(["truck_id", g])[c].transform("size")
        stuck_c = (run >= 5) & df[c].notna()
        stuck |= stuck_c
        df.loc[stuck_c, c] = np.nan
    log["Stuck-sensor readings set to missing"] = int(stuck.sum())
    p = df.payload_t_avg <= 0
    log["Zero payload (scale failure) set to missing"] = int(p.sum())
    df.loc[p, "payload_t_avg"] = np.nan
    log["Remaining missing sensor values (dropouts)"] = int(df[SENS].isna().sum().sum())
    return df, log


def add_maintenance_context(df, mlog, fleet):
    """Hours since last PM, days since last breakdown, truck age and model."""
    mlog = mlog.assign(date=pd.to_datetime(mlog.date))
    out = []
    for tid, g in df.groupby("truck_id", sort=False):
        g = g.copy()
        ev = mlog[mlog.truck_id == tid]
        pm = np.sort(ev.loc[ev.event_type == "PM service", "date"].values)
        bd = np.sort(ev.loc[ev.event_type == "Breakdown", "date"].values)
        d = g.date.values
        cum = g.engine_hours.cumsum().values - g.engine_hours.values  # hours before today
        i = np.searchsorted(pm, d, side="right") - 1
        last_pm = np.where(i >= 0, pm[np.clip(i, 0, None)], np.datetime64("NaT"))
        cum_at_pm = np.array([cum[np.searchsorted(d, x)] if not pd.isna(x) else np.nan for x in last_pm])
        g["hours_since_pm"] = cum - cum_at_pm
        j = np.searchsorted(bd, d, side="left") - 1  # previous breakdown strictly before today
        g["days_since_breakdown"] = np.where(j >= 0, (d - bd[np.clip(j, 0, None)]).astype("timedelta64[D]").astype(float), np.nan)
        k = np.searchsorted(bd, d, side="left")      # next breakdown on or after today
        nxt = np.where(k < len(bd), bd[np.clip(k, 0, len(bd) - 1)], np.datetime64("NaT"))
        g["days_to_breakdown"] = (nxt - d).astype("timedelta64[D]").astype(float)
        g["next_component"] = pd.Series(nxt).map(
            ev[ev.event_type == "Breakdown"].groupby("date").component.first()).values
        out.append(g)
    df = pd.concat(out).merge(fleet[["truck_id", "model", "pit", "commissioned"]], on="truck_id")
    df["age_years"] = (df.date - df.commissioned).dt.days / 365.25
    df["breakdown_today"] = df.days_to_breakdown == 0
    df["y"] = df.days_to_breakdown.between(1, HORIZON).astype(int)
    return df.drop(columns="commissioned")


def build_features(df):
    """Per-truck rolling features on operating days. The baseline uses rows t-37..t-8 so a slow
    drift over the last week is not absorbed into the truck's own 'normal'."""
    df = df.sort_values(["truck_id", "date"]).reset_index(drop=True)
    g = df.groupby("truck_id")
    feats = []
    for c in SENS:
        s = g[c]
        base = s.transform(lambda x: x.shift(8).rolling(30, min_periods=15).median())
        sd = s.transform(lambda x: x.shift(8).rolling(30, min_periods=15).std())
        r3 = s.transform(lambda x: x.rolling(3, min_periods=2).mean())
        r7 = s.transform(lambda x: x.rolling(7, min_periods=4).mean())
        w = np.arange(7) - 3.0  # least-squares slope over a full 7-day window = Σ w·x / Σ w²
        slope = s.transform(lambda x: x.rolling(7).apply(lambda v: w @ v / 28.0, raw=True))
        df[f"{c}__z3"] = SIGN[c] * (r3 - base) / sd
        df[f"{c}__z7"] = SIGN[c] * (r7 - base) / sd
        df[f"{c}__slope7"] = SIGN[c] * slope / sd
        df[f"{c}__dev7"] = SIGN[c] * (r7 - base)
        feats += [f"{c}__z3", f"{c}__z7", f"{c}__slope7", f"{c}__dev7"]
    df["max_z3"] = df[[f"{c}__z3" for c in SENS]].max(axis=1)
    df["n_sensors_z2"] = (df[[f"{c}__z3" for c in SENS]] > 2).sum(axis=1)
    df["load_ratio"] = df.payload_t_avg / df.groupby("model").payload_t_avg.transform("median")
    df["is_belaz"] = (df.model == "BelAZ 7555B").astype(int)
    df["south_pit"] = (df.pit == "South Pit").astype(int)
    ctx = ["hours_since_pm", "days_since_breakdown", "age_years", "ambient_c", "overload_share",
           "load_ratio", "is_belaz", "south_pit", "max_z3", "n_sensors_z2"]
    return df, feats + ctx


def ewma_rule(df, z=3.0):
    """Baseline used by many sites: alert when any sensor's 3-day mean is > z sigma off its own baseline."""
    return (df["max_z3"] > z).astype(int)


def alert_episodes(day, score, thr, cooldown=HORIZON):
    """Turn daily scores into inspection episodes: first alert opens an episode, the truck is not
    re-alerted for `cooldown` days."""
    d = day.assign(score=score.values).loc[lambda x: x.score >= thr].sort_values(["truck_id", "date"])
    eps, last = [], {}
    for r in d.itertuples():
        if r.truck_id in last and (r.date - last[r.truck_id]).days < cooldown:
            continue
        last[r.truck_id] = r.date
        eps.append((r.truck_id, r.date))
    return pd.DataFrame(eps, columns=["truck_id", "date"])


def evaluate(day, score, thr, breakdowns):
    """Event-level evaluation and money. `breakdowns`: rows of the maintenance log inside the window."""
    eps = alert_episodes(day, score, thr)
    bd = breakdowns.copy()
    bd["unplanned_azn"] = bd.cost_azn + bd.downtime_h * MARGIN_PER_H
    bd["planned_azn"] = PLANNED_COST * bd.cost_azn + PLANNED_DOWN * bd.downtime_h * MARGIN_PER_H
    lead = []
    for r in bd.itertuples():
        e = eps[(eps.truck_id == r.truck_id) & ((r.date - eps.date).dt.days.between(1, HORIZON))]
        lead.append((r.date - e.date.min()).days if len(e) else np.nan)
    bd["lead_days"] = lead
    caught = bd.lead_days.notna()
    tp_eps = sum(((bd.truck_id == e.truck_id) & ((bd.date - e.date).dt.days.between(1, HORIZON))).any()
                 for e in eps.itertuples())
    saving = P_FIND * (bd.unplanned_azn - bd.planned_azn)[caught].sum() - len(eps) * INSPECTION_AZN
    return dict(threshold=thr, episodes=len(eps), true_episodes=int(tp_eps),
                precision=tp_eps / max(len(eps), 1), recall=caught.mean(),
                median_lead=float(bd.lead_days.median()) if caught.any() else np.nan,
                net_saving_azn=saving, baseline_cost_azn=bd.unplanned_azn.sum()), bd
