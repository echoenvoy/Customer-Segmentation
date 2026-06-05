"""
store_sql.py
============
Phase 3 - SQL Storage

Loads the cleaned transactions and customer features into a local SQLite
database (data/processed/retail.db) and runs 5 analytical SQL queries.

Tables created:
  transactions   - cleaned transaction rows (from transactions_clean.csv)
  customers      - customer-level feature table (from customer_features.csv)

SQL queries demonstrated:
  Q1  - Top 10 customers by total revenue
  Q2  - Revenue and order count by country
  Q3  - Monthly revenue trend
  Q4  - Average RFM values (will be useful for cluster profiling later)
  Q5  - Customers with high cancellation rate AND high monetary value (risky VIPs)

Run: python src/store_sql.py
"""

import os
import sqlite3
import pandas as pd

# -- Paths ----------------------------------------------------------------------
CLEAN_CSV    = "data/processed/transactions_clean.csv"
FEATURES_CSV = "data/processed/customer_features.csv"
DB_PATH      = "data/processed/retail.db"


def load_to_sqlite(clean: pd.DataFrame, features: pd.DataFrame):
    """Write both DataFrames to SQLite, replacing existing tables."""
    conn = sqlite3.connect(DB_PATH)

    # if_exists='replace' drops and recreates the table on each run
    clean.to_sql("transactions", conn, if_exists="replace", index=False)
    features.to_sql("customers",    conn, if_exists="replace", index=False)

    conn.close()
    print(f"[INFO] Database written to {DB_PATH}")


def run_query(conn, title: str, sql: str):
    """Execute a SQL query, print the title and result."""
    print(f"\n-- {title} {'-' * (54 - len(title))}")
    result = pd.read_sql_query(sql, conn)
    print(result.to_string(index=False))


def run_all_queries():
    """Open a read-only connection and run analytical queries."""
    conn = sqlite3.connect(DB_PATH)

    # Q1 - Top 10 customers by revenue
    run_query(conn, "Q1: Top 10 customers by revenue", """
        SELECT CustomerID,
               ROUND(Monetary, 2)       AS TotalRevenue,
               Frequency                AS Orders,
               ROUND(AvgOrderValue, 2)  AS AvgOrderValue
        FROM   customers
        ORDER  BY Monetary DESC
        LIMIT  10;
    """)

    # Q2 - Revenue and order count by country
    run_query(conn, "Q2: Revenue by country", """
        SELECT Country,
               COUNT(DISTINCT InvoiceNo)    AS TotalOrders,
               ROUND(SUM(Revenue), 2)       AS TotalRevenue,
               COUNT(DISTINCT CustomerID)   AS UniqueCustomers
        FROM   transactions
        GROUP  BY Country
        ORDER  BY TotalRevenue DESC
        LIMIT  10;
    """)

    # Q3 - Monthly revenue trend
    run_query(conn, "Q3: Monthly revenue trend", """
        SELECT SUBSTR(InvoiceDate, 1, 7)   AS YearMonth,
               COUNT(DISTINCT InvoiceNo)   AS Orders,
               ROUND(SUM(Revenue), 2)      AS Revenue
        FROM   transactions
        GROUP  BY YearMonth
        ORDER  BY YearMonth;
    """)

    # Q4 - Average RFM values across all customers
    run_query(conn, "Q4: Average RFM values", """
        SELECT ROUND(AVG(Recency),   1)  AS AvgRecency,
               ROUND(AVG(Frequency), 1)  AS AvgFrequency,
               ROUND(AVG(Monetary),  2)  AS AvgMonetary,
               COUNT(*)                  AS TotalCustomers
        FROM customers;
    """)

    # Q5 - High-cancellation customers who are also high-value (risky VIPs)
    run_query(conn, "Q5: Risky VIPs (high cancel rate + high monetary)", """
        SELECT CustomerID,
               ROUND(Monetary, 2)             AS Revenue,
               Frequency                      AS Orders,
               ROUND(CancellationRate * 100, 1) AS CancelPct
        FROM   customers
        WHERE  CancellationRate > 0.2          -- more than 20 % cancellations
          AND  Monetary > (SELECT AVG(Monetary) FROM customers)
        ORDER  BY CancellationRate DESC
        LIMIT  10;
    """)

    conn.close()


if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)

    clean    = pd.read_csv(CLEAN_CSV, dtype={"CustomerID": str})
    features = pd.read_csv(FEATURES_CSV, dtype={"CustomerID": str})

    load_to_sqlite(clean, features)
    run_all_queries()
