"""Synthetic haul-truck telemetry + maintenance log for an open-pit gold mine in western Azerbaijan.

Each truck has four failure modes (engine, transmission, hydraulics/brakes, suspension). A failure
process starts with a daily hazard that depends on truck age, hours since the last PM service,
overloading and summer heat. Most failures are preceded by a degradation period (5-25 days) in
which specific sensors drift; some are sudden (no precursor). Data issues are injected on purpose.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
OUT = Path(__file__).parent
START, END = pd.Timestamp("2024-11-01"), pd.Timestamp("2026-10-31")
DAYS = pd.date_range(START, END, freq="D")

MODELS = {  # payload class (t), hazard multiplier for hydraulics, fuel base (l/h)
    "CAT 777G": dict(cap=90, hyd=1.00, fuel=98),
    "Komatsu HD785": dict(cap=91, hyd=1.05, fuel=101),
    "BelAZ 7555B": dict(cap=55, hyd=1.30, fuel=72),
}
COMP = {  # base onset rate / truck-year, share sudden, unplanned repair cost (AZN), downtime h
    "engine": dict(rate=1.25, sudden=0.12, cost=42000, down=70),
    "transmission": dict(rate=0.95, sudden=0.10, cost=28000, down=52),
    "hydraulics_brakes": dict(rate=1.9, sudden=0.30, cost=7000, down=14),
    "suspension": dict(rate=1.45, sudden=0.15, cost=11000, down=26),
}
N_TRUCKS = 48

# --- fleet -------------------------------------------------------------------
models = rng.choice(list(MODELS), N_TRUCKS, p=[0.45, 0.25, 0.30])
commissioned = pd.to_datetime(rng.choice(pd.date_range("2016-01-01", "2024-06-30", freq="MS"), N_TRUCKS))
fleet = pd.DataFrame({
    "truck_id": [f"HT-{i:02d}" for i in range(1, N_TRUCKS + 1)],
    "model": models,
    "payload_class_t": [MODELS[m]["cap"] for m in models],
    "commissioned": commissioned.strftime("%Y-%m-%d"),
    "pit": rng.choice(["North Pit", "South Pit"], N_TRUCKS, p=[0.55, 0.45]),
})

doy = DAYS.dayofyear.values
ambient = 10 - 12 * np.cos(2 * np.pi * (doy - 15) / 365.25) + rng.normal(0, 2.5, len(DAYS))

tele, mlog, truth = [], [], []
for k, tr in fleet.iterrows():
    m = MODELS[tr.model]
    age0 = (START - pd.Timestamp(tr.commissioned)).days / 365.25
    off = dict(cool=rng.normal(0, 1.5), oil=rng.normal(0, 15), trans=rng.normal(0, 1.5),
               brake=rng.normal(0, 8), vib=rng.normal(0, 0.3), fuel=rng.normal(0, 3), hyd=rng.normal(0, 4))
    overload_bias = rng.beta(2, 20) + (0.05 if tr.pit == "South Pit" else 0)  # steeper ramps, worse loading
    grade = 1.0 if tr.pit == "North Pit" else 1.25
    pm_target = rng.normal(500, 40)
    hrs_since_pm = rng.uniform(0, 450)
    pending = {}  # comp -> dict(onset, fail, amp)
    down_until = -1
    for t, day in enumerate(DAYS):
        if t <= down_until:
            continue
        age = age0 + t / 365.25
        # planned maintenance (PM) every ~500 engine hours, takes one day; backlog makes some late
        if hrs_since_pm >= pm_target:
            mlog.append(dict(truck_id=tr.truck_id, date=day, event_type="PM service", component="",
                             downtime_h=round(rng.normal(12, 2), 1), cost_azn=round(rng.normal(3200, 300))))
            hrs_since_pm = 0
            pm_target = rng.normal(500, 40) + (rng.exponential(120) if rng.random() < 0.25 else 0)
            continue
        overload = float(np.clip(overload_bias + rng.normal(0, 0.03), 0, 0.6))
        # start new failure processes
        for c, p in COMP.items():
            if c in pending:
                continue
            mult = np.exp(0.06 * (age - 5)) * np.exp(0.002 * max(0, hrs_since_pm - 450))
            if c in ("transmission", "suspension"):
                mult *= np.exp(5 * (overload - 0.08)) * (grade if c == "suspension" else 1)
            if c == "engine":
                mult *= np.exp(0.035 * (ambient[t] - 10))
            if c == "hydraulics_brakes":
                mult *= m["hyd"] * grade
            if rng.random() < p["rate"] * mult / 365.25:
                lead = 0 if rng.random() < p["sudden"] else int(rng.integers(5, 26))
                pending[c] = dict(onset=t, fail=t + lead, amp=rng.uniform(0.4, 1.3))
        r = {c: 0.0 for c in COMP}
        for c, p in pending.items():
            if p["fail"] > p["onset"]:
                r[c] = p["amp"] * min(1.0, (t - p["onset"]) / (p["fail"] - p["onset"])) ** 1.5
        # a failure due on a PM or downtime day happens on the next operating day
        failing = [c for c, p in pending.items() if p["fail"] <= t]
        hours = float(np.clip(rng.normal(19.5, 1.2), 14, 22))
        if failing:
            hours = float(rng.uniform(2, 17))
        load_ratio = float(np.clip(rng.normal(0.96 + overload * 0.5, 0.03), 0.75, 1.25))
        amb = ambient[t]
        tele.append(dict(
            date=day, truck_id=tr.truck_id, engine_hours=round(hours, 1),
            payload_t_avg=round(m["cap"] * load_ratio, 1), overload_share=round(overload, 3),
            ambient_c=round(amb + rng.normal(0, 0.8), 1),
            coolant_temp_c=round(86 + 0.25 * (amb - 10) + 6 * (load_ratio - 0.96) + off["cool"] + rng.normal(0, 1.2) + 9 * r["engine"], 1),
            oil_pressure_kpa=round(385 + off["oil"] - 2.2 * (age - 5) - 0.6 * (amb - 10) + rng.normal(0, 10) - 85 * r["engine"], 1),
            trans_oil_temp_c=round(82 + 0.2 * (amb - 10) + 8 * (load_ratio - 0.96) + off["trans"] + rng.normal(0, 1.5) + 12 * r["transmission"], 1),
            brake_temp_c=round(160 * grade + 0.5 * (amb - 10) + off["brake"] + rng.normal(0, 8) + 45 * r["hydraulics_brakes"], 1),
            hydraulic_pressure_bar=round(210 + off["hyd"] + rng.normal(0, 4.5) - 22 * r["hydraulics_brakes"], 1),
            vibration_rms_mm_s=round(4.0 + 0.05 * age + 0.8 * overload + off["vib"] + rng.normal(0, 0.35) + 1.9 * r["suspension"] + 0.7 * r["transmission"], 2),
            fuel_rate_lph=round(m["fuel"] * load_ratio + off["fuel"] + rng.normal(0, 3) + 7 * r["engine"], 1),
        ))
        hrs_since_pm += hours
        if failing:
            down_h = 0
            for c in failing:
                p = COMP[c]
                dh = float(p["down"] * rng.lognormal(0, 0.4))
                down_h = max(down_h, dh)
                mlog.append(dict(truck_id=tr.truck_id, date=day, event_type="Breakdown", component=c,
                                 downtime_h=round(dh, 1), cost_azn=round(p["cost"] * rng.lognormal(0, 0.35))))
                truth.append(dict(truck_id=tr.truck_id, component=c, onset=str(DAYS[pending[c]["onset"]].date()),
                                  failure=str(day.date()), lead_days=t - pending[c]["onset"],
                                  amplitude=round(pending[c]["amp"], 2)))
                del pending[c]
            down_until = t + int(round(down_h / 22))

tele = pd.DataFrame(tele)
mlog = pd.DataFrame(mlog)

# --- inject data-quality issues ------------------------------------------------
SENS = ["coolant_temp_c", "oil_pressure_kpa", "trans_oil_temp_c", "brake_temp_c",
        "hydraulic_pressure_bar", "vibration_rms_mm_s", "fuel_rate_lph"]
for c in SENS:  # telemetry dropouts
    tele.loc[rng.random(len(tele)) < 0.006, c] = np.nan
f = (tele.truck_id == "HT-17") & tele.date.between("2025-07-01", "2025-07-21")  # firmware sent °F
tele.loc[f, "coolant_temp_c"] = (tele.loc[f, "coolant_temp_c"] * 9 / 5 + 32).round(1)
s = tele.index[(tele.truck_id == "HT-09") & tele.date.between("2026-02-10", "2026-03-16")]  # stuck sensor
tele.loc[s, "oil_pressure_kpa"] = 372.0
tele.loc[rng.choice(tele.index, 40, replace=False), "payload_t_avg"] = 0.0  # weighing failures
ids = rng.choice(tele.index, int(0.01 * len(tele)), replace=False)  # messy truck ids
tele.loc[ids, "truck_id"] = [rng.choice([t.lower(), t.replace("-0", "-"), t + " "]) for t in tele.loc[ids, "truck_id"]]
tele["date"] = tele.date.dt.strftime("%Y-%m-%d")
dd = rng.choice(tele.index, int(0.02 * len(tele)), replace=False)  # some dates exported as dd.mm.yyyy
tele.loc[dd, "date"] = pd.to_datetime(tele.loc[dd, "date"]).dt.strftime("%d.%m.%Y")
tele = pd.concat([tele, tele.sample(150, random_state=1)]).sample(frac=1, random_state=2).reset_index(drop=True)
mlog["date"] = mlog.date.dt.strftime("%Y-%m-%d")

tele.to_csv(OUT / "telemetry_daily.csv", index=False)
mlog.sort_values(["date", "truck_id"]).to_csv(OUT / "maintenance_log.csv", index=False)
fleet.to_csv(OUT / "fleet.csv", index=False)
(OUT / "ground_truth.json").write_text(json.dumps(truth, indent=1))
print(f"telemetry {tele.shape}, maintenance {mlog.shape}, breakdowns {(mlog.event_type == 'Breakdown').sum()}")
print(mlog[mlog.event_type == "Breakdown"].component.value_counts())
