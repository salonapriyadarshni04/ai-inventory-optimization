"""
AI-Driven Inventory Optimization & Demand Forecasting System
for Smart Business Growth   (Python Mini Project)

Modules: data generation -> EDA -> ML demand forecasting -> inventory
optimization (EOQ, safety stock, reorder point, ABC) -> simulation -> report.
Run:  python inventory_system.py          (full pipeline + figures)
      python inventory_system.py --menu   (interactive console menu)
"""
import os, sys, math, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, "figures")
DATA = os.path.join(BASE, "data")
os.makedirs(FIG, exist_ok=True); os.makedirs(DATA, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})

# ----------------------------------------------------------------- 1. DATA
PRODUCTS = {  # name: (base demand, trend/yr, season amp, peak day, price, cost, lead time, order cost)
    "Rice 5kg":      (120, .05, .10, 300, 320, 250, 3, 400),
    "Smartphone":    (18,  .20, .25, 320, 14000, 11500, 7, 2500),
    "Notebook":      (200, .02, .35, 200, 60, 35, 4, 300),
    "Water Bottle":  (90,  .10, .45, 130, 250, 140, 5, 350),
    "Headphones":    (35,  .15, .30, 330, 1800, 1100, 6, 900),
    "Running Shoes": (28,  .08, .20, 100, 2600, 1700, 8, 1200),
    "Winter Jacket": (22,  .05, .85, 20, 3200, 2100, 10, 1500),
    "Laptop Bag":    (40,  .12, .15, 215, 1500, 900, 6, 700),
}
CATEGORY = {"Rice 5kg": "Grocery", "Smartphone": "Electronics", "Notebook": "Stationery",
            "Water Bottle": "Lifestyle", "Headphones": "Electronics", "Running Shoes": "Apparel",
            "Winter Jacket": "Apparel", "Laptop Bag": "Accessories"}

def generate_data(days=730, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    rows = []
    for p, (base, tr, amp, peak, price, cost, lt, oc) in PRODUCTS.items():
        t = np.arange(days)
        weekly = 1 + np.where(dates.dayofweek >= 5, .25, 0) + np.where(dates.dayofweek == 4, .1, 0)
        yearly = 1 + amp * np.cos(2 * np.pi * (dates.dayofyear - peak) / 365)
        promo = (rng.random(days) < .06).astype(int)
        lam = base * (1 + tr * t / 365) * weekly * yearly * (1 + .3 * promo)
        units = rng.poisson(lam * rng.normal(1, .08, days).clip(.6))
        rows.append(pd.DataFrame({"date": dates, "product": p, "category": CATEGORY[p],
                                  "price": price, "unit_cost": cost, "promo": promo, "units": units}))
    return pd.concat(rows, ignore_index=True)

# ------------------------------------------------------------- 2. FORECAST
LAGS = [7, 14, 28]
def _features(series, promo, dates):
    s = pd.Series(series, index=dates)
    X = pd.DataFrame(index=dates)
    for l in LAGS: X[f"lag{l}"] = s.shift(l)
    X["roll7"] = s.shift(7).rolling(7).mean()
    X["roll28"] = s.shift(7).rolling(28).mean()
    X["dow"] = dates.dayofweek
    X["sin_y"] = np.sin(2 * np.pi * dates.dayofyear / 365)
    X["cos_y"] = np.cos(2 * np.pi * dates.dayofyear / 365)
    X["promo"] = promo
    return X

def forecast_product(df_p, horizon):
    """Train on all but last `horizon` days; recursively forecast the holdout."""
    y = df_p["units"].values.astype(float); promo = df_p["promo"].values; dates = pd.DatetimeIndex(df_p["date"])
    n = len(y) - horizon
    X = _features(y, promo, dates)
    tr = X.iloc[:n].dropna(); ytr = y[tr.index.map(dates.get_loc)]
    lvl_tr = tr["roll28"].values
    model = GradientBoostingRegressor(n_estimators=250, max_depth=3, learning_rate=.05, subsample=.8, random_state=1)
    model.fit(tr, ytr / lvl_tr)
    work = y.copy(); work[n:] = np.nan; preds = []
    for i in range(n, len(y)):
        tmp = pd.Series(work[:i + 1]).fillna(0).values
        row = _features(tmp, promo[:i + 1], dates[:i + 1]).iloc[[-1]]
        yhat = max(model.predict(row)[0] * row['roll28'].values[0], 0); work[i] = yhat; preds.append(yhat)
    in_res = ytr - model.predict(tr) * lvl_tr
    return np.array(preds), y[n:], model, tr.columns, in_res

def baselines(df_p, horizon):
    y = df_p["units"].values.astype(float); n = len(y) - horizon
    ma = np.full(horizon, y[n - 28:n].mean())
    naive = np.array([y[n - 7 + (i % 7)] for i in range(horizon)])   # seasonal naive (last week repeated)
    return ma, naive

def metrics(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    return dict(MAE=np.mean(np.abs(a - f)), RMSE=np.sqrt(np.mean((a - f) ** 2)),
                MAPE=np.mean(np.abs(a - f) / np.maximum(a, 1)) * 100)

# ----------------------------------------------------------- 3. OPTIMIZATION
Z = {0.90: 1.2816, 0.95: 1.6449, 0.98: 2.0537, 0.99: 2.3263}
def optimise(name, daily_forecast, residual_sd, service=0.95, hold_rate=.20):
    base, tr, amp, peak, price, cost, lt, S = PRODUCTS[name]
    D = daily_forecast.mean() * 365; H = cost * hold_rate
    eoq = math.sqrt(2 * D * S / H)
    ss = Z[service] * residual_sd * math.sqrt(lt)
    rop = daily_forecast.mean() * lt + ss
    return dict(product=name, annual_demand=D, EOQ=eoq, safety_stock=ss, reorder_point=rop,
                lead_time=lt, order_cost=S, hold_cost=H, orders_per_year=D / eoq)

def abc_analysis(df):
    v = (df.assign(val=df.units * df.unit_cost).groupby("product").val.sum().sort_values(ascending=False))
    share = v.cumsum() / v.sum() * 100
    cls = ["A" if s - (x / v.sum() * 100) < 70 else ("B" if s - (x / v.sum() * 100) < 90 else "C")
           for s, x in zip(share, v)]
    return pd.DataFrame({"consumption_value": v, "cum_pct": share, "class": cls})

def simulate(name, demand, fc, params, policy):
    """Daily (s,Q) simulation with lead time and lost sales. Returns dict of KPIs and inventory path."""
    base, tr, amp, peak, price, cost, lt, S = PRODUCTS[name]; H = cost * .20 / 365
    inv = params["reorder_point"] + params["EOQ"] if policy == "smart" else demand[:28].mean() * 14 + demand[:28].mean() * lt
    pipe = []; log = []; orders = 0; lost = 0; sold = 0; hold = 0
    for d in range(len(demand)):
        inv += sum(q for t, q in pipe if t == d); pipe = [(t, q) for t, q in pipe if t > d]
        s = min(inv, demand[d]); inv -= s; sold += s; lost += demand[d] - s
        if policy == "smart":
            rop = fc[d + 1:d + 1 + lt].sum() + params["safety_stock"] if d + 1 < len(fc) else params["reorder_point"]
            q = params["EOQ"]
        else:
            rop = demand[:28].mean() * lt; q = demand[:28].mean() * 14   # manual rule: 2 weeks of average demand, no forecast, no buffer
        if inv + sum(q_ for _, q_ in pipe) <= rop:
            pipe.append((d + lt, q)); orders += 1
        hold += inv * H; log.append(inv)
    ordc = orders * S; stock_c = lost * (price - cost)
    return dict(orders=orders, lost=lost, fill=sold / (sold + lost) * 100, holding=hold, ordering=ordc,
                stockout=stock_c, total=hold + ordc + stock_c, path=np.array(log))

# ------------------------------------------------------------------ FIGURES
def fig_flow():
    fig, ax = plt.subplots(figsize=(10, 3.4)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 3.4)
    boxes = [("Sales Data\n(CSV / Generated)", .3), ("Data Cleaning\n& EDA", 2.2), ("Feature\nEngineering", 4.1),
             ("ML Demand\nForecast", 6.0), ("Inventory\nOptimizer", 7.9)]
    for t, x in boxes:
        ax.add_patch(plt.Rectangle((x, 1.7), 1.6, 1.1, fc="#dbe9f7", ec="#1f4e79", lw=1.5)); ax.text(x + .8, 2.25, t, ha="center", va="center", fontsize=9)
    for i in range(4): ax.annotate("", (boxes[i + 1][1], 2.25), (boxes[i][1] + 1.6, 2.25), arrowprops=dict(arrowstyle="->", lw=1.5))
    outs = [("Reorder Alerts", 4.1), ("Dashboards\n& Graphs", 6.0), ("Cost Simulation\n& KPIs", 7.9)]
    for t, x in outs:
        ax.add_patch(plt.Rectangle((x, .2), 1.6, .9, fc="#e2f0d9", ec="#38761d", lw=1.5)); ax.text(x + .8, .65, t, ha="center", va="center", fontsize=9)
        ax.annotate("", (x + .8, 1.1), (7.9 + .8 if x == 7.9 else 7.9 + .8, 1.7), arrowprops=dict(arrowstyle="->", lw=1.2, connectionstyle="arc3,rad=0"))
    fig.savefig(f"{FIG}/fig01_flow_diagram.png", dpi=200, bbox_inches="tight"); plt.close()

def figures(df, res, abc, opt, sims):
    tot = df.groupby("date").units.sum()
    fig, ax = plt.subplots(figsize=(9, 3.6)); ax.plot(tot.index, tot, lw=.6, alpha=.5, label="Daily")
    ax.plot(tot.index, tot.rolling(30).mean(), lw=2, color="crimson", label="30-day average")
    ax.set(title="Total Daily Units Sold (All Products)", ylabel="Units"); ax.legend(); fig.savefig(f"{FIG}/fig02_sales_trend.png", dpi=200, bbox_inches="tight"); plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    d = df.assign(dow=df.date.dt.day_name().str[:3]).groupby("dow").units.mean().reindex(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    axs[0].bar(d.index, d.values, color="#2e75b6"); axs[0].set(title="Average Units by Weekday", ylabel="Units/day")
    m = df.groupby(df.date.dt.month).units.mean(); axs[1].plot(m.index, m.values, marker="o", color="#c55a11")
    axs[1].set(title="Average Units by Month", xlabel="Month"); fig.savefig(f"{FIG}/fig03_seasonality.png", dpi=200, bbox_inches="tight"); plt.close()

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    for ax, p in zip(axs.ravel(), ["Winter Jacket", "Water Bottle", "Smartphone", "Rice 5kg"]):
        r = res[p]; x = np.arange(len(r["actual"]))
        ax.plot(x, r["actual"], label="Actual", lw=1.4); ax.plot(x, r["pred"], label="Forecast (GBM)", lw=1.6, ls="--", color="crimson")
        ax.set(title=f"{p}  (MAPE {r['m_ml']['MAPE']:.1f}%)", xlabel="Day of test period", ylabel="Units"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig04_forecast_vs_actual.png", dpi=200); plt.close()

    comp = pd.DataFrame({"Moving Average": [r["m_ma"]["MAPE"] for r in res.values()],
                         "Seasonal Naive": [r["m_nv"]["MAPE"] for r in res.values()],
                         "ML (Gradient Boosting)": [r["m_ml"]["MAPE"] for r in res.values()]}, index=res.keys())
    ax = comp.plot.bar(figsize=(10, 3.8), color=["#a6a6a6", "#8faadc", "#c00000"], width=.8)
    ax.set(ylabel="MAPE (%)  - lower is better", title="Forecast Accuracy Comparison"); plt.xticks(rotation=25, ha="right")
    plt.savefig(f"{FIG}/fig05_model_comparison.png", dpi=200, bbox_inches="tight"); plt.close()

    fig, ax = plt.subplots(figsize=(8, 3.8)); cols = {"A": "#c00000", "B": "#ed7d31", "C": "#70ad47"}
    ax.bar(abc.index, abc.consumption_value / 1e6, color=[cols[c] for c in abc["class"]]); ax.set_ylabel("Consumption value (Rs. million)")
    ax2 = ax.twinx(); ax2.plot(abc.index, abc.cum_pct, color="k", marker="o"); ax2.set_ylabel("Cumulative %"); ax2.grid(False); ax2.axhline(80, ls=":", color="gray")
    ax.set_title("ABC (Pareto) Analysis of Inventory"); plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.savefig(f"{FIG}/fig06_abc_pareto.png", dpi=200, bbox_inches="tight"); plt.close()

    o = opt.set_index("product").loc["Smartphone"]; Q = np.linspace(5, 4 * o.EOQ, 200)
    h = Q / 2 * o.hold_cost; s = o.annual_demand / Q * o.order_cost
    fig, ax = plt.subplots(figsize=(7.5, 3.8)); ax.plot(Q, h, label="Holding cost"); ax.plot(Q, s, label="Ordering cost"); ax.plot(Q, h + s, lw=2.5, color="k", label="Total cost")
    ax.axvline(o.EOQ, color="crimson", ls="--", label=f"EOQ = {o.EOQ:.0f} units"); ax.set(xlabel="Order quantity", ylabel="Annual cost (Rs.)", title="EOQ Cost Curve - Smartphone"); ax.legend()
    fig.savefig(f"{FIG}/fig07_eoq_curve.png", dpi=200, bbox_inches="tight"); plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, pol, t in zip(axs, ["manual", "smart"], ["Manual rule (no forecast)", "AI-driven policy"]):
        pth = sims["Water Bottle"][pol]["path"]; ax.plot(pth, color="#2e75b6"); ax.axhline(0, color="red", lw=1)
        ax.set(title=t + " - Water Bottle", xlabel="Day", ylabel="Stock on hand"); ax.set_ylim(bottom=-10)
    fig.savefig(f"{FIG}/fig08_inventory_levels.png", dpi=200, bbox_inches="tight"); plt.close()

    k = pd.DataFrame({pol: [sims[p][pol]["total"] / 1e3 for p in sims] for pol in ["manual", "smart"]}, index=list(sims))
    ax = k.plot.bar(figsize=(10, 3.8), color=["#a6a6a6", "#2e75b6"], width=.75); ax.set_ylabel("Total cost (Rs. thousand)")
    ax.set_title("Total Inventory Cost: Manual vs AI-driven Policy"); ax.legend(["Manual", "AI-driven"]); plt.xticks(rotation=25, ha="right")
    plt.savefig(f"{FIG}/fig09_cost_comparison.png", dpi=200, bbox_inches="tight"); plt.close()

    fig, ax = plt.subplots(figsize=(7, 3.6)); fr = pd.DataFrame({"Manual": [sims[p]["manual"]["fill"] for p in sims], "AI-driven": [sims[p]["smart"]["fill"] for p in sims]}, index=list(sims))
    fr.plot.barh(ax=ax, color=["#a6a6a6", "#38761d"]); ax.set(xlabel="Fill rate (%)", xlim=(70, 101), title="Service Level (Fill Rate)")
    fig.savefig(f"{FIG}/fig10_fill_rate.png", dpi=200, bbox_inches="tight"); plt.close()

# --------------------------------------------------------------- PIPELINE
def run(horizon=60, service=.95, save=True):
    df = generate_data(); df.to_csv(f"{DATA}/sales_data.csv", index=False)
    res, opts, sims = {}, [], {}
    for p, g in df.groupby("product"):
        g = g.reset_index(drop=True)
        pred, act, model, cols, inres = forecast_product(g, horizon); ma, nv = baselines(g, horizon)
        res[p] = dict(pred=pred, actual=act, m_ml=metrics(act, pred), m_ma=metrics(act, ma), m_nv=metrics(act, nv),
                      imp=dict(zip(cols, model.feature_importances_)))
        bt_pred, bt_act, *_ = forecast_product(g.iloc[:len(g) - horizon].reset_index(drop=True), 30)   # out-of-sample backtest
        o = optimise(p, pred, float(np.sqrt(np.mean((bt_act - bt_pred) ** 2))), service); opts.append(o)
        sims[p] = {pol: simulate(p, act, pred, o, pol) for pol in ["manual", "smart"]}
    opt = pd.DataFrame(opts); abc = abc_analysis(df)
    opt["class"] = opt["product"].map(abc["class"])
    if save:
        figures(df, res, abc, opt, sims); fig_flow()
        opt.round(1).to_csv(f"{DATA}/inventory_policy.csv", index=False)
    return df, res, opt, abc, sims

def summary(res, opt, abc, sims):
    acc = pd.DataFrame({p: {k: r[k] for k in []} | {"ML MAPE%": r["m_ml"]["MAPE"], "MA MAPE%": r["m_ma"]["MAPE"],
                       "Naive MAPE%": r["m_nv"]["MAPE"], "ML RMSE": r["m_ml"]["RMSE"]} for p, r in res.items()}).T
    cost = pd.DataFrame({p: {"Manual cost": s["manual"]["total"], "AI cost": s["smart"]["total"],
                             "Manual fill%": s["manual"]["fill"], "AI fill%": s["smart"]["fill"],
                             "Manual lost units": s["manual"]["lost"], "AI lost units": s["smart"]["lost"]} for p, s in sims.items()}).T
    return acc, cost

def menu():
    df, res, opt, abc, sims = run()
    acc, cost = summary(res, opt, abc, sims)
    while True:
        print("\n===== AI INVENTORY & DEMAND FORECASTING SYSTEM =====\n1. Forecast accuracy\n2. Reorder policy (EOQ / SS / ROP)\n"
              "3. ABC analysis\n4. Cost & service comparison\n5. Stock check for a product\n6. Exit")
        c = input("Choice: ").strip()
        if c == "1": print(acc.round(2))
        elif c == "2": print(opt.round(1).to_string(index=False))
        elif c == "3": print(abc.round(1))
        elif c == "4": print(cost.round(1)); print(f"\nTotal saving: Rs. {cost['Manual cost'].sum() - cost['AI cost'].sum():,.0f}")
        elif c == "5":
            p = input("Product name: ").strip(); q = float(input("Current stock: "))
            if p in opt["product"].values:
                r = opt.set_index("product").loc[p]
                print("REORDER NOW - order %d units" % round(r.EOQ) if q <= r.reorder_point else "Stock OK (reorder point %.0f)" % r.reorder_point)
            else: print("Unknown product.")
        elif c == "6": break

if __name__ == "__main__":
    if "--menu" in sys.argv: menu()
    else:
        df, res, opt, abc, sims = run(); acc, cost = summary(res, opt, abc, sims)
        print(acc.round(2)); print(opt.round(1).to_string(index=False)); print(abc.round(1)); print(cost.round(1))
        print("TOTAL manual:", cost["Manual cost"].sum(), "AI:", cost["AI cost"].sum())
