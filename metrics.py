"""Cleaning + KPI functions for Nassau Candy profitability analysis."""
import numpy as np
import pandas as pd

DIVISION_FIX = {"chocolate": "Chocolate", "sugar": "Sugar", "other": "Other"}


def load_and_clean(path):
    """Load raw CSV and apply the cleaning steps from the project brief."""
    df = pd.read_csv(path)
    n_raw = len(df)

    # Standardize labels (trim, collapse spaces, fix dash spacing, consistent case)
    df["Product Name"] = (df["Product Name"].str.strip()
                          .str.replace(r"\s+", " ", regex=True)
                          .str.replace(r"\s*-\s*", " - ", regex=True))
    df["Division"] = (df["Division"].str.strip().str.lower()
                      .map(DIVISION_FIX).fillna(df["Division"].str.strip().str.title()))

    # Dates are dd-mm-yyyy
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d-%m-%Y", errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d-%m-%Y", errors="coerce")

    # Missing units: impute from sales / median unit price of the product, min 1
    df["Units"] = pd.to_numeric(df["Units"], errors="coerce")
    unit_price = (df["Sales"] / df["Units"]).groupby(df["Product ID"]).transform("median")
    est = (df["Sales"] / unit_price).round()
    df["Units"] = df["Units"].fillna(est).clip(lower=1)

    # Validate: remove zero/negative sales, missing/invalid cost or profit, no order date
    df = df.dropna(subset=["Sales", "Cost", "Gross Profit", "Order Date"])
    df = df[(df["Sales"] > 0) & (df["Cost"] >= 0)]
    # Gross Profit must equal Sales - Cost (tolerance 1 cent)
    df = df[(df["Sales"] - df["Cost"] - df["Gross Profit"]).abs() <= 0.01]

    df = df.drop_duplicates(subset="Row ID").copy()
    df["Month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
    df["Margin %"] = df["Gross Profit"] / df["Sales"] * 100
    df.attrs["rows_removed"] = n_raw - len(df)
    return df


def product_metrics(df):
    """Per-product KPIs: margin, profit/unit, contributions, volatility."""
    g = df.groupby(["Product Name", "Division"], as_index=False).agg(
        Sales=("Sales", "sum"), Cost=("Cost", "sum"),
        Profit=("Gross Profit", "sum"), Units=("Units", "sum"),
        Orders=("Order ID", "nunique"))
    g["Gross Margin %"] = g["Profit"] / g["Sales"] * 100
    g["Profit per Unit"] = g["Profit"] / g["Units"]
    g["Cost Ratio %"] = g["Cost"] / g["Sales"] * 100
    g["Revenue Contribution %"] = g["Sales"] / g["Sales"].sum() * 100
    g["Profit Contribution %"] = g["Profit"] / g["Profit"].sum() * 100

    # Margin volatility = std dev of monthly gross margin (percentage points)
    m = df.groupby(["Product Name", "Month"]).agg(S=("Sales", "sum"), P=("Gross Profit", "sum"))
    m["gm"] = m["P"] / m["S"] * 100
    vol = m.groupby("Product Name")["gm"].std().rename("Margin Volatility")
    g = g.merge(vol, on="Product Name", how="left")
    g["Margin Volatility"] = g["Margin Volatility"].fillna(0)
    return g.sort_values("Profit", ascending=False).reset_index(drop=True)


def add_quadrant(p, gm_cut=60.0):
    """Sales vs margin quadrants.
    High sales  = revenue share >= half of the average product share (100/n/2 %).
    High margin = gross margin >= gm_cut (the dashboard margin threshold)."""
    hs_cut = 100 / len(p) / 2
    def q(r):
        hs, hm = r["Revenue Contribution %"] >= hs_cut, r["Gross Margin %"] >= gm_cut
        return ("Star (high sales / high margin)" if hs and hm else
                "Volume trap (high sales / low margin)" if hs else
                "Niche gem (low sales / high margin)" if hm else
                "Laggard (low sales / low margin)")
    p = p.copy()
    p["Quadrant"] = p.apply(q, axis=1)
    return p


def add_flags(p, margin_threshold, low_rev_pct=1.0, watch_band=5.0):
    """Action flags for the cost-structure diagnostics."""
    p = p.copy()
    def f(r):
        gm = r["Gross Margin %"]
        if gm < margin_threshold:
            return ("Discontinuation review" if r["Revenue Contribution %"] < low_rev_pct
                    else "Repricing needed")
        if gm < margin_threshold + watch_band:
            return "Cost renegotiation watch"
        return "Healthy"
    p["Action Flag"] = p.apply(f, axis=1)
    return p


def division_metrics(df):
    d = df.groupby("Division", as_index=False).agg(
        Sales=("Sales", "sum"), Cost=("Cost", "sum"),
        Profit=("Gross Profit", "sum"), Units=("Units", "sum"),
        Products=("Product Name", "nunique"))
    d["Gross Margin %"] = d["Profit"] / d["Sales"] * 100
    d["Revenue Share %"] = d["Sales"] / d["Sales"].sum() * 100
    d["Profit Share %"] = d["Profit"] / d["Profit"].sum() * 100
    d["Profit - Revenue Gap (pp)"] = d["Profit Share %"] - d["Revenue Share %"]
    return d.sort_values("Profit", ascending=False)


def pareto(p, col):
    """Cumulative share curve + % of products needed to reach 80%."""
    s = p.sort_values(col, ascending=False)[["Product Name", col]].reset_index(drop=True)
    s["Cum %"] = s[col].cumsum() / s[col].sum() * 100
    s["Product Rank %"] = (s.index + 1) / len(s) * 100
    n80 = int((s["Cum %"] < 80).sum() + 1)
    return s, n80, n80 / len(s) * 100


def dependency(p):
    """Concentration indicators: top-1 / top-3 share and HHI of profit."""
    sh = p["Profit"] / p["Profit"].sum()
    top = sh.sort_values(ascending=False)
    hhi = float((sh ** 2).sum() * 10000)
    return {"Top-1 profit share %": top.iloc[0] * 100,
            "Top-3 profit share %": top.head(3).sum() * 100,
            "HHI (0-10000)": hhi}
