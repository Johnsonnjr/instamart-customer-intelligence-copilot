import os
import sqlite3
import pandas as pd
import numpy as np

RAW_DIR = "data/raw"
CLEAN_DIR = "data/clean"
DB_PATH = "data/instamart.db"

os.makedirs(CLEAN_DIR, exist_ok=True)

def audit():
    customers = pd.read_csv(f"{RAW_DIR}/customers.csv")
    products = pd.read_csv(f"{RAW_DIR}/products.csv")
    orders = pd.read_csv(f"{RAW_DIR}/orders.csv")

    tables = {
        "CUSTOMERS": customers,
        "PRODUCTS": products,
        "ORDERS": orders
    }

    # Shape, data types and missing-value report
    for name, df in tables.items():
        print(f"\n===== {name} AUDIT =====")
        print("Shape:", df.shape)

        print("\nData types:")
        print(df.dtypes)

        missing = pd.DataFrame({
            "missing_count": df.isna().sum(),
            "missing_percentage": (df.isna().mean() * 100).round(2)
        })

        print("\nMissing-value report:")
        print(missing)

    # Specific audit figures
    amount = pd.to_numeric(orders["order_amount"], errors="coerce")

    age = pd.to_numeric(customers["age"], errors="coerce")
    gender = customers["gender"].dropna().astype(str).str.strip()
    gender = gender[gender != ""]

    print("\n===== REQUIRED AUDIT FIGURES =====")
    print("Raw order rows:", len(orders))
    print("Exact duplicate rows:", orders.duplicated().sum())
    print("Pending orders:", (orders["status"] == "pending").sum())
    print("Distinct order IDs:", orders["order_id"].nunique())
    print("Non-numeric order amounts:", amount.isna().sum())
    print("Negative delivery times:", (orders["delivery_minutes"] < 0).sum())
    print("Blank delivery times:", orders["delivery_minutes"].isna().sum())

    print("Blank customer ages:", age.isna().sum())
    print(
        "Ages outside 10-100:",
        (age.notna() & ~age.between(10, 100)).sum()
    )
    print("Distinct raw city spellings:", customers["city"].nunique())
    print("Distinct non-blank gender spellings:", gender.nunique())
    print("Blank genders:", customers["gender"].isna().sum())

def clean_data():
    customers = pd.read_csv(f"{RAW_DIR}/customers.csv")

        # Standardize city names
    customers["city"] = (customers["city"].str.strip().str.title())
    print("City counts:")
    print(customers["city"].value_counts())




        # Standardize gender values
    customers["gender"] = (
        customers["gender"]
        .fillna("")
        .str.strip()
        .str.lower()
    )

    gender_map = {
        "m": "male",
        "male": "male",
        "f": "female",
        "female": "female"
    }

    customers["gender"] = (
        customers["gender"]
        .map(gender_map)
        .fillna("unknown")
    )

    print("\nGender counts:")
    print(customers["gender"].value_counts())


        # Clean invalid ages
    customers["age"] = pd.to_numeric(
        customers["age"], errors="coerce"
    )

    customers.loc[
        ~customers["age"].between(10, 100),
        "age"
    ] = np.nan

    # Fill missing ages using city-wise median
    customers["age"] = customers["age"].fillna(
        customers.groupby("city")["age"].transform("median")
    )

    print("\nAge validation:")
    print("Missing ages:", customers["age"].isna().sum())
    print("Minimum age:", customers["age"].min())
    print("Maximum age:", customers["age"].max())

        # Convert signup_date to datetime
    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"],
        errors="coerce"
    )

    print("\nSignup date data type:")
    print(customers["signup_date"].dtype)

        # Load raw orders
    orders = pd.read_csv(f"{RAW_DIR}/orders.csv")

    # Convert order timestamp
    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"], errors="coerce"
    )

    print("\nOrders timestamp data type:")
    print(orders["order_ts"].dtype)

        # Remove exact duplicate rows
    before = len(orders)

    orders = orders.drop_duplicates().copy()

    after = len(orders)

    print("\nDuplicate removal:")
    print("Rows before:", before)
    print("Rows after:", after)
    print("Duplicates removed:", before - after)

        # Keep the latest record for each order_id
    orders = orders.sort_values("order_ts")

    orders = orders.drop_duplicates(
        subset=["order_id"],
        keep="last"
    ).copy()

    print("\nLatest order validation:")
    print("Total rows:", len(orders))
    print("Distinct order IDs:", orders["order_id"].nunique())    # Convert order_amount to numeric
    orders["order_amount"] = pd.to_numeric(
        orders["order_amount"],
        errors="coerce"
    )

    print("\nOrder amount validation:")
    print("Missing amounts:", orders["order_amount"].isna().sum())


    products = pd.read_csv(f"{RAW_DIR}/products.csv")

    print("\nOrder columns:")
    print(orders.columns.tolist())

    print("\nProduct columns:")
    print(products.columns.tolist())

        # Merge product prices into orders
    orders = orders.merge(
        products[["product_id", "price_inr"]],
        on="product_id",
        how="left"
    )

    # Calculate missing order amounts
    calculated_amount = (
        orders["price_inr"]
        * orders["quantity"]
        * (1 - orders["discount_pct"] / 100)
    ).round(2)

    orders["order_amount"] = orders["order_amount"].fillna(
        calculated_amount
    )

    # Remove temporary price column
    orders = orders.drop(columns=["price_inr"])

    print("\nFinal amount validation:")
    print("Missing amounts:", orders["order_amount"].isna().sum())


        # Convert negative delivery times to missing values
    orders.loc[
        orders["delivery_minutes"] < 0,
        "delivery_minutes"
    ] = np.nan

    print("\nDelivery time validation:")
    print(
        "Negative delivery times:",
        (orders["delivery_minutes"] < 0).sum()
    )
    print(
        "Missing delivery times:",
        orders["delivery_minutes"].isna().sum()
    )

        # Merge product category into orders
    orders = orders.merge(
        products[["product_id", "category"]],
        on="product_id",
        how="left"
    )

    # Calculate category-wise median using completed orders
    completed_mask = orders["status"].eq("completed")

    category_median = (
        orders.loc[completed_mask]
        .groupby("category")["delivery_minutes"]
        .transform("median")
    )

    # Fill missing delivery times for completed orders only
    orders.loc[completed_mask, "delivery_minutes"] = (
        orders.loc[completed_mask, "delivery_minutes"]
        .fillna(category_median)
    )

    print("\nDelivery time final validation:")
    print(
        "Completed missing:",
        orders.loc[completed_mask, "delivery_minutes"].isna().sum()
    )
    print(
        "Cancelled missing:",
        orders.loc[~completed_mask, "delivery_minutes"].isna().sum()
    )

    print("\nOrder status counts:")
    print(orders["status"].value_counts())

    print("\nPending orders:")
    print((orders["status"] == "pending").sum())


        # Save cleaned CSV files
    customers.to_csv(
        f"{CLEAN_DIR}/customers.csv", index=False
    )

    products.to_csv(
        f"{CLEAN_DIR}/products.csv", index=False
    )

    orders.to_csv(
        f"{CLEAN_DIR}/orders.csv", index=False
    )

    print("\nClean CSV files saved successfully!")


        # Create SQLite database
    conn = sqlite3.connect(DB_PATH)

    customers.to_sql(
        "customers", conn, if_exists="replace", index=False
    )

    products.to_sql(
        "products", conn, if_exists="replace", index=False
    )

    orders.to_sql(
        "orders", conn, if_exists="replace", index=False
    )

    conn.close()

    print("SQLite database created successfully!")

def validate_data():
    customers = pd.read_csv("data/clean/customers.csv")
    products = pd.read_csv("data/clean/products.csv")
    orders = pd.read_csv("data/clean/orders.csv")

    print("Customers:", len(customers))
    print("Products:", len(products))
    print("Orders:", len(orders))
    print("Unique Order IDs:", orders["order_id"].nunique())
    print("Missing Order Amounts:", orders["order_amount"].isna().sum())
    completed_mask = orders["status"] == "completed"
    print("Missing Delivery (Completed):",
    orders.loc[completed_mask, "delivery_minutes"].isna().sum()
)

    cancelled_mask = orders["status"] == "cancelled"
    print("Missing Delivery (Cancelled):",
    orders.loc[cancelled_mask, "delivery_minutes"].isna().sum()
)
    print("\nOrder Status Counts:")
    print(orders["status"].value_counts())
    print(
    "Completed Order Amount:",
    round(
        orders.loc[completed_mask, "order_amount"].sum(),
        2
    )

)
    print(
    "Missing Customer Ages:",
    customers["age"].isna().sum()
)

    print("\nCustomer City Counts:")
    print(customers["city"].value_counts())
        # Actual validation checks
    assert len(customers) == 1200
    assert len(products) == 40
    assert len(orders) == 12309

    assert orders["order_id"].nunique() == 12309
    assert orders["order_amount"].isna().sum() == 0

    assert orders.loc[completed_mask, "delivery_minutes"].isna().sum() == 0
    assert orders.loc[cancelled_mask, "delivery_minutes"].isna().sum() == 1199

    assert (orders["status"] == "completed").sum() == 11110
    assert (orders["status"] == "cancelled").sum() == 1199

    assert round(orders.loc[completed_mask, "order_amount"].sum(), 2) == 2998668.80
    assert customers["age"].isna().sum() == 0

    print("\nALL CHECKPOINTS PASSED")