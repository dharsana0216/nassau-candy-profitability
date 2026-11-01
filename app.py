import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import metrics as m

st.set_page_config(page_title="Nassau Candy | Profitability", page_icon="🍬", layout="wide")


@st.cache_data
def get_data():
    return m.load_and_clean("Nassau_Candy_Distributor.csv")


raw = get_data()

# ---------------- Sidebar filters ----------------
st.sidebar.title("🍬 Filters")
dmin, dmax = raw["Order Date"].min().date(), raw["Order Date"].max().date()
dr = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
if isinstance(dr, tuple) and len(dr) == 2:
    d0, d1 = dr
else:
    d0, d1 = dmin, dmax
divs = st.sidebar.multiselect("Division", sorted(raw["Division"].unique()),
                              default=sorted(raw["Division"].unique()))
threshold = st.sidebar.slider("Margin threshold (%)", 0, 90, 60, 1,
                              help="Products below this gross margin are flagged as risk.")
search = st.sidebar.text_input("Product search", "")

df = raw[(raw["Order Date"].dt.date >= d0) & (raw["Order Date"].dt.date <= d1)
         & (raw["Division"].isin(divs))]
if search:
    df = df[df["Product Name"].str.contains(search, case=False, na=False)]

st.title("Nassau Candy: Product Line Profitability & Margin Performance")
if df.empty:
    st.warning("No data for the selected filters.")
    st.stop()

prod = m.add_flags(m.add_quadrant(m.product_metrics(df), threshold), threshold)

# ---------------- KPI strip ----------------
tot_s, tot_p = df["Sales"].sum(), df["Gross Profit"].sum()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Revenue", f"${tot_s:,.0f}")
c2.metric("Gross profit", f"${tot_p:,.0f}")
c3.metric("Gross margin", f"{tot_p / tot_s * 100:.1f}%")
c4.metric("Profit / unit", f"${tot_p / df['Units'].sum():.2f}")
c5.metric("Products below threshold", int((prod["Gross Margin %"] < threshold).sum()))

t1, t2, t3, t4 = st.tabs(["Product Profitability", "Division Performance",
                          "Cost vs Margin Diagnostics", "Profit Concentration"])

# ---------------- 1. Product overview ----------------
with t1:
    a, b = st.columns(2)
    rank_by = a.radio("Rank by", ["Profit", "Gross Margin %"], horizontal=True)
    lb = prod.sort_values(rank_by, ascending=False)
    fig = px.bar(lb, x=rank_by, y="Product Name", orientation="h", color="Division",
                 title=f"Product leaderboard by {rank_by}")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
    a.plotly_chart(fig, use_container_width=True)

    fig = px.pie(prod, names="Product Name", values="Profit", hole=0.45,
                 title="Profit contribution")
    b.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(prod, x="Sales", y="Gross Margin %", size="Profit", color="Quadrant",
                     hover_name="Product Name", log_x=True,
                     title="Sales vs margin (bubble = profit)")
    fig.add_hline(y=threshold, line_dash="dash", annotation_text="margin threshold")
    st.plotly_chart(fig, use_container_width=True)

    cols = ["Product Name", "Division", "Sales", "Profit", "Units", "Gross Margin %",
            "Profit per Unit", "Revenue Contribution %", "Profit Contribution %",
            "Margin Volatility", "Quadrant"]
    st.dataframe(lb[cols].round(2), use_container_width=True, hide_index=True)
    st.caption("Margin volatility = std. dev. of monthly gross margin (pp). It is ~0 here because "
               "each product sells at a fixed price/cost, so margin differences are between products, "
               "not over time.")

# ---------------- 2. Division ----------------
with t2:
    dv = m.division_metrics(df)
    a, b = st.columns(2)
    melt = dv.melt(id_vars="Division", value_vars=["Sales", "Profit"], var_name="Metric")
    a.plotly_chart(px.bar(melt, x="Division", y="value", color="Metric", barmode="group",
                          title="Revenue vs profit by division"), use_container_width=True)
    b.plotly_chart(px.bar(dv, x="Division", y="Gross Margin %", color="Division", text_auto=".1f",
                          title="Average gross margin by division"), use_container_width=True)
    fig = px.box(df, x="Division", y="Margin %", color="Division", points=False,
                 title="Order-level margin distribution by division")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(dv.round(2), use_container_width=True, hide_index=True)
    weak = dv[dv["Gross Margin %"] < threshold]
    if len(weak):
        st.error("Structural margin issue: " + ", ".join(weak["Division"]) +
                 f" (below {threshold}% threshold)")
    else:
        st.success("All selected divisions are above the margin threshold.")

# ---------------- 3. Cost vs margin ----------------
with t3:
    fig = px.scatter(prod, x="Sales", y="Cost", size="Units", color="Action Flag",
                     hover_name="Product Name", log_x=True, log_y=True,
                     title="Cost vs sales (above the diagonal = cost exceeds sales)")
    lim = [min(prod["Sales"].min(), prod["Cost"].min()), max(prod["Sales"].max(), prod["Cost"].max())]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(dash="dot", color="grey"),
                             name="Break-even"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Margin risk flags")
    flag_cols = ["Product Name", "Division", "Sales", "Cost Ratio %", "Gross Margin %",
                 "Profit per Unit", "Units", "Action Flag"]
    risk = prod[prod["Action Flag"] != "Healthy"].sort_values("Gross Margin %")
    st.dataframe(risk[flag_cols].round(2), use_container_width=True, hide_index=True)
    st.caption("Rules: below threshold & <1% of revenue → Discontinuation review; below threshold & "
               "≥1% revenue → Repricing; within 5 pp above threshold → Cost renegotiation watch.")

# ---------------- 4. Pareto ----------------
with t4:
    a, b = st.columns(2)
    out = {}
    for col, label, container in [("Sales", "Revenue", a), ("Profit", "Profit", b)]:
        s, n80, pct = m.pareto(prod, col)
        out[label] = (n80, pct)
        fig = go.Figure()
        fig.add_bar(x=s["Product Name"], y=s[col], name=label)
        fig.add_scatter(x=s["Product Name"], y=s["Cum %"], name="Cumulative %",
                        yaxis="y2", mode="lines+markers")
        fig.update_layout(title=f"Pareto: {label}", yaxis2=dict(overlaying="y", side="right",
                          range=[0, 105], title="Cumulative %"), xaxis_tickangle=-40,
                          legend=dict(orientation="h"))
        fig.add_hline(y=80, line_dash="dash", yref="y2")
        container.plotly_chart(fig, use_container_width=True)
    k1, k2 = st.columns(2)
    k1.metric("Products for 80% of revenue", f"{out['Revenue'][0]} ({out['Revenue'][1]:.0f}% of products)")
    k2.metric("Products for 80% of profit", f"{out['Profit'][0]} ({out['Profit'][1]:.0f}% of products)")

    dep = m.dependency(prod)
    d1_, d2_, d3_ = st.columns(3)
    d1_.metric("Top-1 product profit share", f"{dep['Top-1 profit share %']:.1f}%")
    d2_.metric("Top-3 product profit share", f"{dep['Top-3 profit share %']:.1f}%")
    d3_.metric("HHI (profit)", f"{dep['HHI (0-10000)']:.0f}")
    if dep["Top-3 profit share %"] > 50:
        st.warning("Dependency risk: three products generate over half of gross profit.")

    reg = df.groupby("Region", as_index=False)["Gross Profit"].sum()
    st.plotly_chart(px.bar(reg, x="Region", y="Gross Profit", title="Profit by region (dependency check)"),
                    use_container_width=True)

st.sidebar.download_button("⬇ Download product KPIs (CSV)",
                           prod.round(3).to_csv(index=False), "product_kpis.csv")
