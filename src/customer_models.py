import joblib
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

ORDERS_PATH = "data/clean/orders.csv"
CUSTOMERS_PATH = "data/clean/customers.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_rfm():
    orders = pd.read_csv(ORDERS_PATH)
    customers = pd.read_csv(CUSTOMERS_PATH)

    orders["order_ts"] = pd.to_datetime(orders["order_ts"])

    completed_orders = orders[
        orders["status"] == "completed"
    ]
    snapshot_date = pd.Timestamp("2025-10-01 00:00:00")


    rfm = completed_orders.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max")
    ).reset_index()

    rfm["recency"] = (
        snapshot_date - rfm["last_order_ts"]
    ).dt.days

    frequency = completed_orders.groupby("customer_id").agg(
        frequency=("order_id", "nunique")
    ).reset_index()

    rfm = rfm.merge(frequency, on="customer_id")

    monetary = completed_orders.groupby("customer_id").agg(
        monetary=("order_amount", "sum")
    ).reset_index()

    rfm = rfm.merge(monetary, on="customer_id")

    rfm_features = rfm[["recency", "frequency", "monetary"]]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(rfm_features)

    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(scaled_features)

    print("\nPCA Explained Variance Ratio:")
    print(pca.explained_variance_ratio_)
    print("Total variance explained:", pca.explained_variance_ratio_.sum())

    for k in range(2, 9):
        kmeans = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=42
        )
        kmeans.fit(scaled_features)

        score = silhouette_score(
            scaled_features,
            kmeans.labels_
        )

        print(f"K={k}, Inertia={kmeans.inertia_:.1f}, "f"Silhouette={score:.3f}")


    final_kmeans = KMeans(
        n_clusters=4,
        n_init=10,
        random_state=42
    )

    rfm["segment"] = final_kmeans.fit_predict(
        scaled_features
    )
    segment_profile = rfm.groupby("segment").agg(
        customer_count=("customer_id", "count"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean")
    ).round(2)

    segment_names = {
        0: "Occasional Customers",
        1: "Premium Customers",
        2: "Loyal Customers",
        3: "At-Risk Customers"
    }

    rfm["segment_name"] = rfm["segment"].map(segment_names)
    rfm["segment_id"] = rfm["segment"]

    rfm["is_premium"] = (
        rfm["segment_name"] == "Premium Customers"
    )

    customer_segments = rfm[
        [
            "customer_id",
            "recency",
            "frequency",
            "monetary",
            "segment_id",
            "segment_name",
            "is_premium"
        ]
    ]

    print("\nCustomer Segment Profile:")
    print(segment_profile)


    customer_segments.to_csv(
        os.path.join(OUTPUT_DIR, "customer_segments.csv"),
        index=False
    )

    print("\nCustomer segments saved successfully!")
    print("\nRFM file saved successfully!")

    print("\nRFM Preview:")
    print(rfm.head())

    print("\nRFM Shape:", rfm.shape)

    print("\nRFM Reference Checks:")
    print("Customers:", len(rfm))
    print("Recency median:", rfm["recency"].median())
    print("Recency min:", rfm["recency"].min())
    print("Recency max:", rfm["recency"].max())
    print("Frequency mean:", round(rfm["frequency"].mean(), 2))
    print("Frequency max:", rfm["frequency"].max())
    print("Monetary mean:", round(rfm["monetary"].mean(), 2))
    print("Monetary max:", round(rfm["monetary"].max(), 2))


    print("Completed orders:", len(completed_orders))
    print("Orders shape:", orders.shape)
    print("Customers shape:", customers.shape)


def create_lapse_dataset():

    orders = pd.read_csv(ORDERS_PATH)
    customers = pd.read_csv(CUSTOMERS_PATH)

    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    customers["signup_date"] = pd.to_datetime(customers["signup_date"])

    cutoff_date = pd.Timestamp("2025-09-01")
    label_end_date = pd.Timestamp("2025-10-01")


        # Customer tenure in days
    customers["tenure_days"] = (
        cutoff_date - customers["signup_date"]
    ).dt.days


    historical_orders = orders[
        (orders["order_ts"] < cutoff_date) &
        (orders["status"] == "completed")
    ].copy()


    september_orders = orders[
        (orders["order_ts"] >= cutoff_date) &
        (orders["order_ts"] < label_end_date) &
        (orders["status"] == "completed")
    ].copy()

        # Customers with at least one completed order before cutoff
    cohort = historical_orders[
        ["customer_id"]
    ].drop_duplicates()

    # September completed-order customer IDs
    september_customer_ids = set(
        september_orders["customer_id"]
    )

    cohort["lapsed"] = (
        ~cohort["customer_id"].isin(september_customer_ids)
    ).astype(int)

    frequency = historical_orders.groupby("customer_id").agg(
        frequency=("order_id", "nunique")
    ).reset_index()

    monetary = historical_orders.groupby("customer_id").agg(
        monetary=("order_amount", "sum")
    ).reset_index()


    last_order = historical_orders.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max")
    ).reset_index()

    last_order["recency"] = (
        cutoff_date - last_order["last_order_ts"]
    ).dt.days

    print("\nRecency Preview:")
    print(last_order.head())

    avg_discount = historical_orders.groupby("customer_id").agg(
        avg_discount=("discount_pct", "mean")
    ).reset_index()


    favourite_category = historical_orders.groupby("customer_id").agg(
        favourite_category=(
            "category",
            lambda x: x.mode().iloc[0]
        )
    ).reset_index()

    final_status_orders = orders[
        (orders["order_ts"] < cutoff_date) &
        (orders["status"].isin(["completed", "cancelled"]))
    ]

    # Customer-wise cancellation rate
    cancellation_rate = final_status_orders.groupby("customer_id").agg(
        total_orders=("order_id", "nunique"),
        cancelled_orders=(
            "status",
            lambda x: (x == "cancelled").sum()
        )
    ).reset_index()

    cancellation_rate["cancellation_rate"] = (
        cancellation_rate["cancelled_orders"] /
        cancellation_rate["total_orders"]
    )

    avg_delivery = historical_orders.groupby("customer_id").agg(
        avg_delivery_minutes=("delivery_minutes", "mean")
    ).reset_index()

        # Combine all customer features
    features = cohort.copy()

    features = features.merge(frequency, on="customer_id", how="left")
    features = features.merge(monetary, on="customer_id", how="left")

    features = features.merge(
        last_order[["customer_id", "recency"]],
        on="customer_id",
        how="left"
    )

    features = features.merge(avg_discount, on="customer_id", how="left")

    features = features.merge(
        cancellation_rate[["customer_id", "cancellation_rate"]],
        on="customer_id",
        how="left"
    )

    features = features.merge(avg_delivery, on="customer_id", how="left")

    features = features.merge(
        favourite_category,
        on="customer_id",
        how="left"
    )

    # Add customer demographic features
    customer_features = customers[
        ["customer_id", "city", "age", "gender", "tenure_days"]
    ]

    features = features.merge(
        customer_features,
        on="customer_id",
        how="left"
    )

    print("\nCombined Features Preview:")
    print(features.head())

    print("\nFeatures Shape:", features.shape)

    print("\nMissing Values Check:")
    print(features.isna().sum())

    print(
        "\nTotal Missing Values:",
        features.drop(
            columns=["customer_id", "lapsed"]
        ).isna().sum().sum()
    )

        # Separate features and target
    X = features.drop(
        columns=["customer_id", "lapsed"]
    )

    y = features["lapsed"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print("\nTrain-Test Split:")
    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("\nTraining label distribution:")
    print(y_train.value_counts())
    print("\nTesting label distribution:")
    print(y_test.value_counts())



    numerical_features = [
        "frequency",
        "monetary",
        "recency",
        "avg_discount",
        "cancellation_rate",
        "avg_delivery_minutes",
        "age",
        "tenure_days"
    ]

    categorical_features = [
        "favourite_category",
        "city",
        "gender"
    ]

        # Numerical preprocessing
    numerical_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])

    # Combine both transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

        # Fit preprocessing only on training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # Transform test data without fitting
    X_test_processed = preprocessor.transform(X_test)

    print("\nPreprocessing Results:")
    print("Training shape:", X_train_processed.shape)
    print("Testing shape:", X_test_processed.shape)


        # Compare training scaler mean with full cohort mean
    frequency_index = numerical_features.index("frequency")

    training_frequency_mean = (
        preprocessor
        .named_transformers_["num"]
        .named_steps["scaler"]
        .mean_[frequency_index]
    )

    full_cohort_frequency_mean = features["frequency"].mean()

    print("\nFrequency Mean Comparison:")
    print("Training scaler mean:", training_frequency_mean)
    print("Full cohort mean:", full_cohort_frequency_mean)


    logistic_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ])

        # Logistic Regression hyperparameter tuning
    logistic_params = {
        "model__C": [0.01, 0.1, 1, 10, 100]
    }

    logistic_grid = GridSearchCV(
        logistic_pipeline,
        logistic_params,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    logistic_grid.fit(X_train, y_train)

    print("\nLogistic Regression Results:")
    print("Best Parameters:", logistic_grid.best_params_)
    print("Best CV F1 Score:", logistic_grid.best_score_)

        # Random Forest pipeline
    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])
        # Random Forest hyperparameter tuning
    rf_params = {
        "model__max_depth": [3, 5, 8],
        "model__min_samples_leaf": [5, 10, 20]
    }

    rf_grid = GridSearchCV(
        rf_pipeline,
        rf_params,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    rf_grid.fit(X_train, y_train)

    print("\nRandom Forest Results:")
    print("Best Parameters:", rf_grid.best_params_)
    print("Best CV F1 Score:", rf_grid.best_score_)

        # Predict using both best models
    logistic_predictions = logistic_grid.predict(X_test)
    rf_predictions = rf_grid.predict(X_test)

    # Logistic Regression evaluation
    print("\nLogistic Regression Test Results:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, logistic_predictions))

    print("Precision:", precision_score(
        y_test, logistic_predictions, zero_division=0
    ))
    print("Recall:", recall_score(
        y_test, logistic_predictions, zero_division=0
    ))
    print("F1 Score:", f1_score(
        y_test, logistic_predictions, zero_division=0
    ))

    # Random Forest evaluation
    print("\nRandom Forest Test Results:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, rf_predictions))

    print("Precision:", precision_score(
        y_test, rf_predictions, zero_division=0
    ))
    print("Recall:", recall_score(
        y_test, rf_predictions, zero_division=0
    ))
    print("F1 Score:", f1_score(
        y_test, rf_predictions, zero_division=0
    ))

        # Oversampling experiment - training data only
    training_data = X_train.copy()
    training_data["lapsed"] = y_train.values

    majority_class = training_data[
        training_data["lapsed"] == 0
    ]

    minority_class = training_data[
        training_data["lapsed"] == 1
    ]

    minority_oversampled = minority_class.sample(
        n=len(majority_class),
        replace=True,
        random_state=42
    )

    balanced_training = pd.concat(
        [majority_class, minority_oversampled]
    ).sample(frac=1, random_state=42)

    X_train_balanced = balanced_training.drop(columns=["lapsed"])
    y_train_balanced = balanced_training["lapsed"]

    print("\nBalanced Training Distribution:")
    print(y_train_balanced.value_counts())

        # Train Logistic Regression on balanced data
    balanced_logistic = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=1000,
            C=logistic_grid.best_params_["model__C"]
        ))
    ])

    balanced_logistic.fit(
        X_train_balanced,
        y_train_balanced
    )

    # Predict on unchanged test data
    balanced_predictions = balanced_logistic.predict(X_test)

    print("\nOversampled Logistic Regression Results:")
    print("Precision:", precision_score(
        y_test, balanced_predictions, zero_division=0
    ))
    print("Recall:", recall_score(
        y_test, balanced_predictions, zero_division=0
    ))
    print("F1 Score:", f1_score(
        y_test, balanced_predictions, zero_division=0
    ))

        # Random Forest - Top 5 Feature Importances
    rf_model = rf_grid.best_estimator_.named_steps["model"]

    feature_names = (
        rf_grid.best_estimator_
        .named_steps["preprocessor"]
        .get_feature_names_out()
    )

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": rf_model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    print("\nTop 5 Random Forest Features:")
    print(importance_df.head(5))

        # Logistic Regression - Top 3 Coefficients
    lr_model = logistic_grid.best_estimator_.named_steps["model"]

    lr_feature_names = (
        logistic_grid.best_estimator_
        .named_steps["preprocessor"]
        .get_feature_names_out()
    )

    coefficient_df = pd.DataFrame({
        "feature": lr_feature_names,
        "coefficient": lr_model.coef_[0]
    })

    coefficient_df["absolute_value"] = (
        coefficient_df["coefficient"].abs()
    )

    coefficient_df = coefficient_df.sort_values(
        by="absolute_value",
        ascending=False
    )

    print("\nTop 3 Logistic Regression Coefficients:")
    print(coefficient_df.head(3))




    print("\nAverage Delivery Preview:")
    print(avg_delivery.head())

    print("\nCancellation Rate Preview:")
    print(cancellation_rate.head())

    print("\nAverage Discount Preview:")
    print(avg_discount.head())

    print("\nMonetary Preview:")
    print(monetary.head())

    print("\nFrequency Preview:")
    print(frequency.head())
    print("\nLapse Cohort Summary:")
    print("Total customers:", len(cohort))
    print("Lapsed customers:", (cohort["lapsed"] == 1).sum())
    print("Not lapsed customers:", (cohort["lapsed"] == 0).sum())

    print(
        "Lapse percentage:",
        round(cohort["lapsed"].mean() * 100, 2),
        "%"
    )

        # Save the selected Logistic Regression model
    os.makedirs("outputs", exist_ok=True)

    selected_model = logistic_grid.best_estimator_

    joblib.dump(
        selected_model,
        "outputs/lapse_model.joblib"
    )

    print("\nModel Saved Successfully!")

    # Reload the saved model
    loaded_model = joblib.load(
        "outputs/lapse_model.joblib"
    )

    # Compare predictions
    original_predictions = selected_model.predict(X_test)
    loaded_predictions = loaded_model.predict(X_test)

    predictions_match = np.array_equal(
        original_predictions,
        loaded_predictions
    )

    print("Predictions Match:", predictions_match)


        # Create Premium Customer Retention List

    # Get test customer IDs
    test_customer_ids = features.loc[
        X_test.index, "customer_id"
    ].values

    # Predict lapse probabilities
    lapse_probabilities = selected_model.predict_proba(
        X_test
    )[:, 1]

    # Create retention candidates
    retention_candidates = pd.DataFrame({
        "customer_id": test_customer_ids,
        "lapse_probability": lapse_probabilities,
        "actually_lapsed": y_test.values
    })

    # Load customer segments
    segments = pd.read_csv(
        "outputs/customer_segments.csv"
    )

    # Select premium customers only
    premium_customers = segments[
        segments["is_premium"] == True
    ][["customer_id"]]

    # Match premium customers with test predictions
    retention_list = retention_candidates.merge(
        premium_customers,
        on="customer_id",
        how="inner"
    )

    # Rank by highest lapse probability
    retention_list = retention_list.sort_values(
        by="lapse_probability",
        ascending=False
    ).head(10)

    # Save retention list
    retention_list.to_csv(
        "outputs/retention_list.csv",
        index=False
    )

    # Count actually lapsed customers
    true_lapsed_count = (
        retention_list["actually_lapsed"] == 1
    ).sum()

    print("\nPremium Customer Retention List:")
    print(retention_list)

    print("Actually lapsed in Top 10:", true_lapsed_count)



if __name__ == "__main__":
    create_rfm()
