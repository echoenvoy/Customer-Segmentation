import os
import numpy as np
import pandas as pd

np.random.seed(42)
os.makedirs("data/raw", exist_ok=True)

N_CUSTOMERS  = 4_000
N_ROWS       = 500_000
N_PRODUCTS   = 3_500

# Random products
products = {
    f"P{i:05d}": (
        np.random.choice(["MUG", "CANDLE", "FRAME", "BAG", "TOY", "CLOCK"]) + f" {i}",
        round(np.random.lognormal(1.5, 0.7), 2)  # price
    )
    for i in range(N_PRODUCTS)
}
stock_codes = list(products.keys())
descriptions = [products[s][0] for s in stock_codes]
prices       = {s: products[s][1] for s in stock_codes}

# Random transactions
invoice_nos  = [f"INV{np.random.randint(100000,999999)}" for _ in range(N_ROWS)]
customer_ids = np.random.choice(
    [str(10000 + i) for i in range(N_CUSTOMERS)] + [None]*500,  # ~1% missing
    size=N_ROWS
)
stock_choice = np.random.choice(stock_codes, size=N_ROWS)

df = pd.DataFrame({
    "InvoiceNo":   invoice_nos,
    "StockCode":   stock_choice,
    "Description": [products[s][0] for s in stock_choice],
    "Quantity":    np.random.randint(1, 50, size=N_ROWS),
    "InvoiceDate": pd.date_range("2010-12-01", periods=N_ROWS, freq="1min")[:N_ROWS],
    "UnitPrice":   [prices[s] for s in stock_choice],
    "CustomerID":  customer_ids,
    "Country":     np.random.choice(
        ["United Kingdom","Germany","France","Spain","Netherlands","Belgium"],
        size=N_ROWS,
        p=[0.70, 0.10, 0.08, 0.05, 0.04, 0.03]
    ),
})

# Add ~2 % cancellations
cancel_mask = np.random.rand(N_ROWS) < 0.02
df.loc[cancel_mask, "InvoiceNo"] = "C" + df.loc[cancel_mask, "InvoiceNo"]
df.loc[cancel_mask, "Quantity"]  = -df.loc[cancel_mask, "Quantity"]

df.to_csv("data/raw/online_retail_raw.csv", index=False)
print(f"[INFO] Demo dataset saved: {df.shape[0]:,} rows, {df['CustomerID'].nunique():,} unique customers")
