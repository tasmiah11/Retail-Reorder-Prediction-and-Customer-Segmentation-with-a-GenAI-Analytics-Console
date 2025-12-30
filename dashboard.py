import os
import json
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    root_mean_squared_error,
    silhouette_score,
)

from openai import OpenAI

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

OPENAI_MODEL = "gpt-5-mini"  # change if you want


# ----------------------------- DATA PREP -----------------------------


@st.cache_data(show_spinner=True)
def load_base_tables() -> Dict[str, pd.DataFrame]:
    aisles = pd.read_csv("Data/aisles.csv")
    departments = pd.read_csv("Data/departments.csv")
    orders = pd.read_csv("Data/orders.csv")
    op_prior = pd.read_csv("Data/order_products__prior.csv")
    op_train = pd.read_csv("Data/order_products__train.csv")
    products = pd.read_csv("Data/products.csv")

    return {
        "aisles": aisles,
        "departments": departments,
        "orders": orders,
        "op_prior": op_prior,
        "op_train": op_train,
        "products": products,
    }


def remove_outlier_rows_iqr(
    df: pd.DataFrame, column: str, factor: float
) -> pd.DataFrame:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return df[(df[column] >= lower) & (df[column] <= upper)]


@st.cache_data(show_spinner=True)
def build_line_level_df(
    max_users: int = 20000, random_state: int = 4830
) -> pd.DataFrame:
    tables = load_base_tables()
    aisles = tables["aisles"]
    departments = tables["departments"]
    orders = tables["orders"]
    op_prior = tables["op_prior"]
    op_train = tables["op_train"]
    products = tables["products"]

    order_products = pd.concat([op_prior, op_train], ignore_index=True)

    prod_full = (
        products.merge(aisles, on="aisle_id", how="left")
        .merge(departments, on="department_id", how="left")
    )

    data = (
        order_products.merge(orders, on="order_id", how="left")
        .merge(prod_full, on="product_id", how="left")
    )

    # sample users here to shrink everything but keep full history per user
    if max_users is not None:
        user_ids = data["user_id"].dropna().unique()
        if len(user_ids) > max_users:
            rng = np.random.default_rng(random_state)
            sampled_users = rng.choice(user_ids, size=max_users, replace=False)
            data = data[data["user_id"].isin(sampled_users)]

    basket_size = (
        data.groupby("order_id")["product_id"]
        .count()
        .reset_index(name="basket_size")
    )

    data = data.merge(basket_size, on="order_id", how="left")

    cols_with_outliers = ["add_to_cart_order", "order_number", "basket_size"]

    df = data.copy()
    for col in cols_with_outliers:
        df = remove_outlier_rows_iqr(df, col, 3)

    df = df.dropna(
        subset=[
            "reordered",
            "order_dow",
            "order_hour_of_day",
            "days_since_prior_order",
            "department",
            "aisle",
        ]
    )

    return df


@st.cache_data(show_spinner=True)
def build_model_df(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    user_orders = (
        df.groupby("user_id")
        .agg(
            user_n_orders=("order_id", "nunique"),
            user_avg_days_between=("days_since_prior_order", "mean"),
            user_avg_basket_size=("basket_size", "mean"),
        )
        .reset_index()
    )

    prod_stats = (
        df.groupby("product_id")
        .agg(
            prod_n_purchases=("order_id", "count"),
            prod_reorder_rate=("reordered", "mean"),
            prod_avg_add_to_cart=("add_to_cart_order", "mean"),
        )
        .reset_index()
    )

    model_df = (
        df.merge(user_orders, on="user_id", how="left")
        .merge(prod_stats, on="product_id", how="left")
    )

    model_df = model_df[
        [
            "user_id",
            "order_id",
            "product_id",
            "reordered",
            "order_dow",
            "order_hour_of_day",
            "order_number",
            "add_to_cart_order",
            "basket_size",
            "days_since_prior_order",
            "department",
            "aisle",
            "user_n_orders",
            "user_avg_days_between",
            "user_avg_basket_size",
            "prod_n_purchases",
            "prod_reorder_rate",
            "prod_avg_add_to_cart",
        ]
    ].dropna()

    return model_df, user_orders, prod_stats


@st.cache_data(show_spinner=True)
def build_order_level(
    df: pd.DataFrame, user_orders: pd.DataFrame
) -> pd.DataFrame:
    order_level = (
        df.groupby(
            ["order_id", "user_id", "order_dow", "order_hour_of_day", "order_number"]
        )
        .agg(
            basket_size=("product_id", "count"),
            days_since_prior_order=("days_since_prior_order", "max"),
        )
        .reset_index()
    )

    order_level = order_level.merge(user_orders, on="user_id", how="left")
    return order_level


@st.cache_data(show_spinner=True)
def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    cust_base = (
        df.groupby("user_id")
        .agg(
            n_orders=("order_id", "nunique"),
            total_items=("product_id", "count"),
            avg_basket_size=("basket_size", "mean"),
            avg_days_between=("days_since_prior_order", "mean"),
        )
        .reset_index()
    )

    dept_pivot = (
        df.groupby(["user_id", "department"])["product_id"]
        .count()
        .reset_index(name="n_items")
    )

    dept_total = (
        dept_pivot.groupby("user_id")["n_items"]
        .sum()
        .reset_index(name="user_total_items")
    )

    dept_pivot = dept_pivot.merge(dept_total, on="user_id", how="left")
    dept_pivot["dept_share"] = dept_pivot["n_items"] / dept_pivot["user_total_items"]

    dept_wide = (
        dept_pivot.pivot(index="user_id", columns="department", values="dept_share")
        .fillna(0)
    )
    dept_wide.columns = [f"dept_share_{c}" for c in dept_wide.columns]

    cust_features = cust_base.merge(dept_wide, on="user_id", how="left").fillna(0)
    cust_features.set_index("user_id", inplace=True)

    return cust_features


# ---------------------- REORDER CLASSIFICATION ----------------------


@st.cache_resource(show_spinner=True)
def train_reorder_models(model_df: pd.DataFrame) -> Dict[str, Any]:
    X = model_df.drop(columns=["reordered"])
    y = model_df["reordered"]

    numeric_features = [
        "order_dow",
        "order_hour_of_day",
        "order_number",
        "add_to_cart_order",
        "basket_size",
        "days_since_prior_order",
        "user_n_orders",
        "user_avg_days_between",
        "user_avg_basket_size",
        "prod_n_purchases",
        "prod_reorder_rate",
        "prod_avg_add_to_cart",
    ]
    categorical_features = ["department", "aisle"]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.1, random_state=4830, stratify=y
    )

    def evaluate_classifier(name: str, model) -> Tuple[Dict[str, float], Pipeline, np.ndarray]:
        pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        metrics = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "AUC": auc,
            "TN": cm[0, 0],
            "FP": cm[0, 1],
            "FN": cm[1, 0],
            "TP": cm[1, 1],
        }
        return metrics, pipe, y_proba

    results: Dict[str, Any] = {}
    metrics_by_model: Dict[str, Dict[str, float]] = {}
    probas: Dict[str, np.ndarray] = {}
    pipes: Dict[str, Pipeline] = {}

    models = {
        "Logistic (L2)": LogisticRegression(penalty="l2", max_iter=50, n_jobs=-1),
        "Logistic (L1)": LogisticRegression(
            penalty="l1", solver="saga", max_iter=50, n_jobs=-1
        ),
        "Logistic (Elastic Net)": LogisticRegression(
            penalty="elasticnet", l1_ratio=0.5, solver="saga", max_iter=50, n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, min_samples_leaf=50, random_state=4830
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            min_samples_leaf=30,
            n_jobs=-1,
            random_state=4830,
        ),
    }

    for name, model in models.items():
        metrics, pipe, proba = evaluate_classifier(name, model)
        metrics_by_model[name] = metrics
        probas[name] = proba
        pipes[name] = pipe

    # PCA + logistic
    pca_preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("scaler", StandardScaler()),
                        ("pca", PCA(n_components=10)),
                    ]
                ),
                numeric_features,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    pipe_log_pca = Pipeline(
        steps=[
            ("preprocess", pca_preprocess),
            ("model", LogisticRegression(max_iter=50, n_jobs=-1)),
        ]
    )
    pipe_log_pca.fit(X_train, y_train)
    y_proba_pca = pipe_log_pca.predict_proba(X_test)[:, 1]
    metrics_pca = {
        "Accuracy": np.nan,
        "Precision": np.nan,
        "Recall": np.nan,
        "F1": np.nan,
        "AUC": roc_auc_score(y_test, y_proba_pca),
    }
    metrics_by_model["Logistic + PCA"] = metrics_pca
    probas["Logistic + PCA"] = y_proba_pca
    pipes["Logistic + PCA"] = pipe_log_pca

    # use RF as "best" for ranking and promo
    best_name = "Random Forest"
    proba_rf = probas[best_name]

    X_test_with_keys = X_test.copy()
    X_test_with_keys["order_id"] = model_df.loc[X_test.index, "order_id"].values
    X_test_with_keys["reordered_true"] = y_test.values
    X_test_with_keys["proba"] = proba_rf

    def precision_recall_at_k(df: pd.DataFrame, k: int = 5) -> Tuple[float, float]:
        df = df.sort_values(["order_id", "proba"], ascending=[True, False])
        df["rank"] = df.groupby("order_id").cumcount() + 1
        topk = df[df["rank"] <= k]

        tp = (topk["reordered_true"] == 1).sum()
        fp = (topk["reordered_true"] == 0).sum()
        fn = ((df["reordered_true"] == 1) & (df["rank"] > k)).sum()

        precision_k = tp / (tp + fp + 1e-9)
        recall_k = tp / (tp + fn + 1e-9)
        return float(precision_k), float(recall_k)

    prec5, rec5 = precision_recall_at_k(X_test_with_keys, k=5)

    results["metrics_by_model"] = metrics_by_model
    results["pipes"] = pipes
    results["X_test"] = X_test
    results["y_test"] = y_test
    results["X_test_with_keys"] = X_test_with_keys
    results["precision_at_5"] = prec5
    results["recall_at_5"] = rec5
    results["best_model_name"] = best_name

    return results


# ---------------------- BASKET SIZE REGRESSION ----------------------


@st.cache_resource(show_spinner=True)
def train_regression(order_level: pd.DataFrame) -> Dict[str, Any]:
    reg_numeric = [
        "order_dow",
        "order_hour_of_day",
        "order_number",
        "user_n_orders",
        "user_avg_days_between",
        "user_avg_basket_size",
    ]

    X_reg = order_level[reg_numeric]
    y_reg = order_level["basket_size"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    reg_preprocess = Pipeline(steps=[("scaler", StandardScaler())])

    reg_model = Pipeline(
        steps=[("preprocess", reg_preprocess), ("model", Ridge(alpha=1.0))]
    )

    reg_model.fit(X_train, y_train)
    y_pred = reg_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    return {
        "model": reg_model,
        "X_test": X_test,
        "y_test": y_test,
        "mae": mae,
        "rmse": rmse,
    }


# ---------------------- CUSTOMER SEGMENTATION ----------------------


@st.cache_resource(show_spinner=True)
def build_segments(cust_features: pd.DataFrame) -> Dict[str, Any]:
    cust_X = cust_features.copy()

    scaler = StandardScaler()
    cust_scaled = scaler.fit_transform(cust_X)

    pca = PCA(n_components=min(10, cust_scaled.shape[1]))
    cust_pca = pca.fit_transform(cust_scaled)

    rng = np.random.RandomState(42)
    sample_size = min(10000, cust_pca.shape[0])
    idx = rng.choice(cust_pca.shape[0], size=sample_size, replace=False)

    scores = {}
    for k in [3, 4, 5, 6]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_full = km.fit_predict(cust_pca)
        score = silhouette_score(cust_pca[idx], labels_full[idx])
        scores[k] = score

    best_k = max(scores, key=scores.get)
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cust_segments = kmeans.fit_predict(cust_pca)

    cust_features_seg = cust_features.copy()
    cust_features_seg["segment"] = cust_segments

    segment_profile = (
        cust_features_seg.groupby("segment")
        .agg(
            n_customers=("avg_basket_size", "size"),
            avg_basket_size=("avg_basket_size", "mean"),
            avg_days_between=("avg_days_between", "mean"),
            n_orders=("n_orders", "mean"),
        )
        .reset_index()
    )

    dept_cols = [c for c in cust_features_seg.columns if c.startswith("dept_share_")]
    top_depts = (
        cust_features_seg[dept_cols + ["segment"]]
        .reset_index()
        .melt(
            id_vars=["user_id", "segment"],
            value_vars=dept_cols,
            var_name="department",
            value_name="share",
        )
    )
    top_depts["department"] = top_depts["department"].str.replace(
        "dept_share_", "", regex=False
    )

    top_dept_summary = (
        top_depts.groupby(["segment", "department"])["share"]
        .mean()
        .reset_index()
    )
    top_dept_summary = (
        top_dept_summary.sort_values(
            ["segment", "share"], ascending=[True, False]
        )
        .groupby("segment")
        .head(5)
        .reset_index(drop=True)
    )

    return {
        "cust_features": cust_features_seg,
        "segment_profile": segment_profile,
        "top_dept_summary": top_dept_summary,
        "cust_pca": cust_pca,
        "scores": scores,
        "best_k": best_k,
    }


# ---------------------- PROMO SIMULATOR -----------------------------


def build_scored_for_promo(
    reorder_artifacts: Dict[str, Any],
    model_df: pd.DataFrame,
    seg_artifacts: Dict[str, Any],
) -> pd.DataFrame:
    X_test_with_keys = reorder_artifacts["X_test_with_keys"].copy()
    cust_features_seg = seg_artifacts["cust_features"]

    scored = X_test_with_keys.copy()
    scored["user_id"] = model_df.loc[X_test_with_keys.index, "user_id"].values
    scored["product_id"] = model_df.loc[X_test_with_keys.index, "product_id"].values
    scored["department"] = model_df.loc[X_test_with_keys.index, "department"].values

    scored = scored.merge(
        cust_features_seg[["segment"]].reset_index(), on="user_id", how="left"
    )
    return scored


def simulate_promo(
    scored: pd.DataFrame,
    segment_id: int,
    department_name: Optional[str] = None,
    discount_pct: float = 0.10,
    uplift_factor: float = 2.0,
) -> Dict[str, float]:
    df_seg = scored[scored["segment"] == segment_id].copy()
    if department_name:
        df_seg = df_seg[df_seg["department"].str.lower() == department_name.lower()]

    if df_seg.empty:
        return {
            "segment": segment_id,
            "department": department_name,
            "discount_pct": discount_pct,
            "baseline_expected_reorders": 0.0,
            "new_expected_reorders": 0.0,
            "delta_reorders": 0.0,
        }

    baseline = float(df_seg["proba"].sum())
    uplift = 1 + uplift_factor * discount_pct
    df_seg["proba_new"] = (df_seg["proba"] * uplift).clip(0, 0.99)
    new_total = float(df_seg["proba_new"].sum())

    return {
        "segment": int(segment_id),
        "department": department_name,
        "discount_pct": float(discount_pct),
        "baseline_expected_reorders": baseline,
        "new_expected_reorders": new_total,
        "delta_reorders": new_total - baseline,
    }


# ---------------------- GENAI HELPERS -------------------------------


def get_openai_client() -> Optional[OpenAI]:
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None
        client = OpenAI()
        return client
    except Exception:
        return None


ANALYTICS_SYSTEM_PROMPT = """
You translate business questions about grocery customer behavior into pandas code.

You have these DataFrames already in memory:
- df : line level orders and products
- model_df : modelling table with one row per user–order–product
- segment_profile : metrics by customer segment
- top_dept_summary : top departments per segment
- cust_features : customer-level features with a 'segment' column

Rules:
- Return ONLY Python code.
- No comments, no explanation, no markdown.
- Use only pandas and numpy.
- Assume `pd` and `np` are already imported.
- The last line MUST assign the final table to a variable named `result`.
"""


def ask_analytics(
    client: OpenAI,
    question: str,
    df: pd.DataFrame,
    model_df: pd.DataFrame,
    segment_profile: pd.DataFrame,
    top_dept_summary: pd.DataFrame,
    cust_features: pd.DataFrame,
) -> Tuple[Any, str]:
    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=ANALYTICS_SYSTEM_PROMPT,
        input=question,
    )

    code = response.output_text.strip()

    if code.startswith("```"):
        code = code.split("```", 1)[1]
        if "```" in code:
            code = code.split("```", 1)[0]
    code = code.strip()

    local_ns: Dict[str, Any] = {
        "pd": pd,
        "np": np,
        "df": df,
        "model_df": model_df,
        "segment_profile": segment_profile,
        "top_dept_summary": top_dept_summary,
        "cust_features": cust_features,
    }

    try:
        exec(code, {}, local_ns)
    except Exception as e:
        return f"Error running generated code: {e}", code

    result = local_ns.get("result", None)
    return result, code


def explain_segments_for_managers(
    client: OpenAI,
    segment_profile: pd.DataFrame,
    top_dept_summary: pd.DataFrame,
) -> str:
    segment_json = segment_profile.to_dict(orient="records")
    dept_json = top_dept_summary.to_dict(orient="records")

    prompt = f"""
You are a retail analytics assistant.

Here is customer segment performance as JSON:
{json.dumps(segment_json, indent=2)}

Here are the top departments per segment:
{json.dumps(dept_json, indent=2)}

Write a concise non-technical summary for a retail manager:
1. Describe each segment in one or two sentences.
2. Suggest one promotion idea for each segment.
Keep it short and practical.
"""

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
        )
        return response.output_text
    except Exception as e:
        return f"Error generating summary: {e}"


# -------------------------- UI PAGES --------------------------------


def page_eda(df: pd.DataFrame):
    st.header("Exploratory Data Analysis")

    st.write(f"Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}")

    st.dataframe(df.head(20))

    # For plots, optionally downsample if very large
    df_plot = df
    if df.shape[0] > 200_000:
        df_plot = df.sample(n=200_000, random_state=42)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Orders by hour of day")
        hour_counts = df_plot.groupby("order_hour_of_day")["order_id"].count()
        fig, ax = plt.subplots()
        hour_counts.plot(kind="line", marker="o", ax=ax)
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Orders")
        st.pyplot(fig)

        st.subheader("Orders by day of week")
        dow_counts = df_plot.groupby("order_dow")["order_id"].count()
        fig2, ax2 = plt.subplots()
        dow_counts.plot(kind="bar", ax=ax2)
        ax2.set_xlabel("Day of week (0=Sun)")
        ax2.set_ylabel("Orders")
        st.pyplot(fig2)

    with col2:
        st.subheader("Day-of-week × hour-of-day volume")
        dow_hour = (
            df_plot.groupby(["order_dow", "order_hour_of_day"])["order_id"]
            .count()
            .reset_index()
        )
        pivot_dow_hour = dow_hour.pivot(
            index="order_dow", columns="order_hour_of_day", values="order_id"
        )
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        sns.heatmap(pivot_dow_hour, cmap="Blues", ax=ax3)
        ax3.set_xlabel("Hour of day")
        ax3.set_ylabel("Day of week")
        st.pyplot(fig3)

        st.subheader("Reorder rate by department")
        dept_reorder_rate = (
            df_plot.groupby("department")["reordered"].mean().reset_index()
        )
        dept_reorder_rate = dept_reorder_rate.set_index("department")
        fig4, ax4 = plt.subplots(figsize=(4, 6))
        sns.heatmap(
            dept_reorder_rate,
            annot=True,
            fmt=".2f",
            cmap="Reds",
            cbar=False,
            ax=ax4,
        )
        ax4.set_xlabel("Reordered rate")
        ax4.set_ylabel("Department")
        st.pyplot(fig4)


def page_reorder(model_df: pd.DataFrame, reorder_artifacts: Dict[str, Any]):
    st.header("Product Reorder Prediction")

    metrics_df = (
        pd.DataFrame(reorder_artifacts["metrics_by_model"]).T[
            ["Accuracy", "Precision", "Recall", "F1", "AUC"]
        ]
        .sort_values("AUC", ascending=False)
    )

    st.subheader("Model comparison")
    st.dataframe(
        metrics_df.style.format("{:.3f}").highlight_max("AUC", axis=0)
    )

    st.subheader("Top-K business metrics (Random Forest)")
    st.write(
        f"Precision@5: {reorder_artifacts['precision_at_5']:.3f} | "
        f"Recall@5: {reorder_artifacts['recall_at_5']:.3f}"
    )

    st.subheader("Sample scored rows (RF)")
    st.dataframe(reorder_artifacts["X_test_with_keys"].head(20))


def page_regression(order_level: pd.DataFrame, reg_artifacts: Dict[str, Any]):
    st.header("Basket Size Regression")

    st.write(
        f"MAE: {reg_artifacts['mae']:.2f}  |  RMSE: {reg_artifacts['rmse']:.2f}"
    )

    st.subheader("Try a scenario")
    col1, col2, col3 = st.columns(3)
    with col1:
        order_dow = st.slider("Day of week (0=Sun)", 0, 6, 0)
        order_hour = st.slider("Hour of day", 0, 23, 10)
    with col2:
        order_number = st.slider("Customer order number", 1, 100, 10)
        user_n_orders = st.slider("User n_orders", 1, 100, 10)
    with col3:
        avg_days_between = st.slider("User avg days between", 1.0, 30.0, 7.0)
        avg_basket_size = st.slider("User avg basket size", 1.0, 40.0, 10.0)

    if st.button("Predict basket size"):
        X_new = pd.DataFrame(
            [
                {
                    "order_dow": order_dow,
                    "order_hour_of_day": order_hour,
                    "order_number": order_number,
                    "user_n_orders": user_n_orders,
                    "user_avg_days_between": avg_days_between,
                    "user_avg_basket_size": avg_basket_size,
                }
            ]
        )
        y_hat = reg_artifacts["model"].predict(X_new)[0]
        st.success(f"Predicted basket size: {y_hat:.1f} items")


def page_segments(seg_artifacts: Dict[str, Any]):
    st.header("Customer Segmentation")

    st.subheader("Segment profile")
    st.dataframe(seg_artifacts["segment_profile"])

    st.subheader("Top departments per segment")
    st.dataframe(seg_artifacts["top_dept_summary"])

    st.subheader("Cluster quality (silhouette)")
    scores = seg_artifacts["scores"]
    s_df = (
        pd.DataFrame(
            {"k": list(scores.keys()), "silhouette": list(scores.values())}
        )
        .sort_values("k")
        .reset_index(drop=True)
    )
    fig, ax = plt.subplots()
    ax.plot(s_df["k"], s_df["silhouette"], marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette")
    st.pyplot(fig)


def page_promo(
    seg_artifacts: Dict[str, Any],
    reorder_artifacts: Dict[str, Any],
    model_df: pd.DataFrame,
):
    st.header("Promotion impact simulator")

    scored = build_scored_for_promo(reorder_artifacts, model_df, seg_artifacts)

    segments = sorted(scored["segment"].dropna().unique())
    departments = sorted(scored["department"].dropna().unique())

    col1, col2 = st.columns(2)
    with col1:
        segment_id = st.selectbox("Segment", segments, index=0)
        dept_name = st.selectbox(
            "Department (optional)", ["<All>"] + departments, index=0
        )
        if dept_name == "<All>":
            dept_name = None
    with col2:
        discount = st.slider("Discount %", 0.0, 0.5, 0.15, step=0.01)
        uplift_factor = st.slider("Uplift factor", 0.0, 5.0, 2.0, step=0.1)

    if st.button("Simulate"):
        out = simulate_promo(
            scored, int(segment_id), dept_name, discount, uplift_factor
        )
        st.subheader("Expected change in reorders (test set only)")
        st.json(out)


def page_genai(
    df: pd.DataFrame,
    model_df: pd.DataFrame,
    seg_artifacts: Dict[str, Any],
):
    st.header("GenAI analytics and summaries")

    client = get_openai_client()
    if client is None:
        st.warning(
            "Set the OPENAI_API_KEY environment variable to enable GenAI features."
        )
        return

    segment_profile = seg_artifacts["segment_profile"]
    top_dept_summary = seg_artifacts["top_dept_summary"]
    cust_features = seg_artifacts["cust_features"]

    tab1, tab2 = st.tabs(["Manager summary", "Analytics console"])

    with tab1:
        if st.button("Generate manager summary"):
            summary = explain_segments_for_managers(
                client, segment_profile, top_dept_summary
            )
            if summary.startswith("Error generating summary:"):
                st.error(summary)
            else:
                st.text_area("Summary", summary, height=300)

    with tab2:
        question = st.text_input(
            "Ask a question about customers/products/orders",
            value="Show the top 10 departments by total orders and reorder rate.",
        )
        show_code = st.checkbox("Show generated code", value=True)

        if st.button("Run query"):
            with st.spinner("Calling model and running code..."):
                result, code = ask_analytics(
                    client,
                    question,
                    df,
                    model_df,
                    segment_profile,
                    top_dept_summary,
                    cust_features,
                )
            if show_code:
                st.subheader("Generated pandas code")
                st.code(code, language="python")
            st.subheader("Result")
            if isinstance(result, (pd.DataFrame, pd.Series)):
                st.dataframe(result)
            elif isinstance(result, str) and result.startswith(
                "Error running generated code:"
            ):
                st.error(result)
            else:
                st.write(result)


# ----------------------------- MAIN ---------------------------------


def main():
    st.set_page_config(
        page_title="Instacart Retail Analytics",
        layout="wide",
    )

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "EDA",
            "Reorder model",
            "Basket regression",
            "Segments",
            "Promo simulator",
            "GenAI",
        ],
    )

    max_users = st.sidebar.number_input(
        "Max users to sample",
        min_value=1000,
        max_value=100000,
        value=20000,
        step=1000,
    )

    df = build_line_level_df(max_users=int(max_users))
    model_df, user_orders, _ = build_model_df(df)
    order_level = build_order_level(df, user_orders)
    cust_features = build_customer_features(df)

    reorder_artifacts = None
    reg_artifacts = None
    seg_artifacts = None

    if page in ["Reorder model", "Promo simulator"]:
        reorder_artifacts = train_reorder_models(model_df)

    if page == "Basket regression":
        reg_artifacts = train_regression(order_level)

    if page in ["Segments", "Promo simulator", "GenAI"]:
        seg_artifacts = build_segments(cust_features)

    if page == "EDA":
        page_eda(df)
    elif page == "Reorder model":
        page_reorder(model_df, reorder_artifacts)
    elif page == "Basket regression":
        page_regression(order_level, reg_artifacts)
    elif page == "Segments":
        page_segments(seg_artifacts)
    elif page == "Promo simulator":
        page_promo(seg_artifacts, reorder_artifacts, model_df)
    elif page == "GenAI":
        page_genai(df, model_df, seg_artifacts)


if __name__ == "__main__":
    main()
