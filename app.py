import json
import os
from typing import Dict, List, Tuple
import io
import zipfile

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt

import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ARTIFACTS_DIR = os.path.join("model", "artifacts")
DATA_DIR = os.path.join("data")
FEATURES_PATH = os.path.join(DATA_DIR, "feature_names.json")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics_summary.csv")
SAMPLE_TEST_PATH = os.path.join("data", "test_sample.csv")
SMALL_TXT = 8

class ModelBundle:
    def __init__(self, estimator, scaler, feature_names, name):
        self.estimator = estimator
        self.scaler = scaler
        self.feature_names = feature_names
        self.name = name

    def _transform(self, X: pd.DataFrame):
        if self.scaler is None:
            return X[self.feature_names]
        return self.scaler.transform(X[self.feature_names])

    def predict(self, X: pd.DataFrame):
        X_in = self._transform(X)
        y = self.estimator.predict(X_in)
        try:
            if self.name == "XGBoost" and np.min(y) == 0:
                return y + 1
        except Exception:
            pass
        return y

    def predict_proba(self, X: pd.DataFrame):
        if not hasattr(self.estimator, "predict_proba"):
            return None
        X_in = self._transform(X)
        return self.estimator.predict_proba(X_in)

@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame:
    if os.path.exists(METRICS_PATH):
        return pd.read_csv(METRICS_PATH)
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_feature_info() -> Dict:
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"feature_names": [], "target_name": "target"}

@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    model_path = os.path.join(ARTIFACTS_DIR, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    obj = joblib.load(model_path)
    if isinstance(obj, dict) and "estimator" in obj:
        est = obj["estimator"]
        scaler = obj.get("scaler")
        feats = obj.get("feature_names") or load_feature_info().get("feature_names", [])
        return ModelBundle(est, scaler, feats, model_name)
    return obj

def validate_and_split(df: pd.DataFrame, feature_names: List[str], target_name: str) -> Tuple[pd.DataFrame, pd.Series | None]:
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(
            f"Uploaded data missing required feature columns: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    X = df[feature_names].copy()
    y = None
    if target_name in df.columns:
        y = df[target_name].copy()
    return X, y

def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray | None) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["Accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["Precision"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    metrics["Recall"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    metrics["F1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    metrics["MCC"] = float(matthews_corrcoef(y_true, y_pred))
    if y_proba is not None:
        try:
            if y_proba.ndim == 1:
                metrics["AUC"] = float(roc_auc_score(y_true, y_proba))
            else:
                metrics["AUC"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
        except Exception:
            metrics["AUC"] = float("nan")
    else:
        metrics["AUC"] = float("nan")
    return metrics

def annotate_bars(ax, fmt="{:.0f}", pad=2, fontsize=8, expand_ylim=True):
    heights = [p.get_height() for p in ax.patches if not np.isnan(p.get_height())]
    if heights:
        max_h = max(heights)
        y0, y1 = ax.get_ylim()
        if expand_ylim and max_h >= y1 * 0.98:
            ax.set_ylim(y0, max_h * 1.05)
    for p in ax.patches:
        h = p.get_height()
        if np.isnan(h):
            continue
        x = p.get_x() + p.get_width() / 2.0
        ax.annotate(fmt.format(h),
                    (x, h),
                    ha="center", va="bottom",
                    xytext=(0, pad), textcoords="offset points",
                    fontsize=fontsize,
                    clip_on=False)

def plot_confusion(y_true: pd.Series, y_pred: np.ndarray, labels: List[str] | None = None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    if labels:
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels, rotation=0)
    st.pyplot(fig)

def classification_report_tables(y_true: pd.Series, y_pred: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    classes = sorted([k for k in rep.keys() if k not in ("accuracy", "macro avg", "weighted avg")],
                     key=lambda x: int(x) if str(x).isdigit() else x)
    class_rows = []
    for cls in classes:
        d = rep[cls]
        class_rows.append({
            "Class": str(cls),
            "Precision": d.get("precision", np.nan),
            "Recall": d.get("recall", np.nan),
            "F1": d.get("f1-score", np.nan),
            "Support": int(d.get("support", 0)),
        })
    class_df = pd.DataFrame(class_rows)

    summary_rows = []
    if "macro avg" in rep:
        d = rep["macro avg"]
        summary_rows.append({"Average": "macro avg", "Precision": d["precision"], "Recall": d["recall"], "F1": d["f1-score"]})
    if "weighted avg" in rep:
        d = rep["weighted avg"]
        summary_rows.append({"Average": "weighted avg", "Precision": d["precision"], "Recall": d["recall"], "F1": d["f1-score"]})
    summary_df = pd.DataFrame(summary_rows)

    acc_val = rep.get("accuracy", np.nan)
    accuracy_df = pd.DataFrame([{"Accuracy": acc_val}])

    return class_df, summary_df, accuracy_df

def parse_class_mapping(info: Dict) -> Tuple[Dict[int, int] | None, Dict[int, int] | None]:
    mapping = info.get("class_mapping")
    if not mapping:
        return None, None
    to_index = mapping.get("to_index", {})
    to_label = mapping.get("to_label", {})
    try:
        class_to_idx = {int(k): int(v) for k, v in to_index.items()}
        idx_to_class = {int(k): int(v) for k, v in to_label.items()}
        return class_to_idx, idx_to_class
    except Exception:
        return None, None

def normalize_for_confusion(y_true: pd.Series, y_pred: np.ndarray) -> Tuple[pd.Series, pd.Series, List[str]]:
    yt = pd.Series(y_true).astype(str)
    yp = pd.Series(y_pred).astype(str)

    def sort_key(x: str):
        try:
            return (0, int(x))
        except Exception:
            return (1, x)

    labels = sorted(pd.unique(yt), key=sort_key)
    return yt, yp, labels

def map_series_with_dict(s: pd.Series, mapper: Dict[int, int]) -> pd.Series:
    return s.map(lambda v: mapper.get(int(v)) if pd.notna(v) else v)


def map_array_with_dict(arr: np.ndarray, mapper: Dict[int, int]) -> np.ndarray:
    mapped = []
    for v in arr:
        try:
            mapped.append(mapper.get(int(v), v))
        except Exception:
            mapped.append(v)
    return np.array(mapped)


def plotly_confusion_matrix(cm: np.ndarray, labels: List[str], title: str):
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        )
    )
    fig.update_traces(
        text=cm,
        texttemplate="%{text}",
        textfont=dict(size=14),
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=12)),
        width=450,
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(title="Predicted", tickfont=dict(size=14)),
        yaxis=dict(title="Actual", tickfont=dict(size=14)),
    )
    return fig


def main():
    st.set_page_config(page_title="ML Classification Model Explorer", layout="wide")
    st.title("Fetal health classifier")
    st.caption("Upload test CSV, select models, and compare metrics.")
    base_models = ["LogisticRegression", "DecisionTree", "KNN", "GaussianNB", "RandomForest", "XGBoost"]

    metrics_df = load_metrics()
    feature_info = load_feature_info()
    feature_names: List[str] = feature_info.get("feature_names", [])
    target_name: str = feature_info.get("target_name", "target")
    class_to_idx, idx_to_class = parse_class_mapping(feature_info)

    if "select_all_models" not in st.session_state:
        st.session_state.select_all_models = True
    for m in base_models:
        key = f"chk_{m}"
        if key not in st.session_state:
            st.session_state[key] = True

    def apply_select_all():
        sel = st.session_state.select_all_models
        for mm in base_models:
            st.session_state[f"chk_{mm}"] = sel

    with st.sidebar:
        st.header("Controls")
        use_sample = st.toggle("Use sample test data", value=True, help="Load the prepared sample test split.")

        uploaded = st.file_uploader("Upload test CSV", type=["csv"])

        if os.path.exists(SAMPLE_TEST_PATH):
            with open(SAMPLE_TEST_PATH, "rb") as f:
                st.download_button(
                    "Download sample dataset",
                    data=f.read(),
                    file_name=os.path.basename(SAMPLE_TEST_PATH),
                    mime="text/csv",
                )
        else:
            st.caption("Sample dataset not found.")

        st.markdown("Models")
        st.checkbox("Select all models", key="select_all_models", on_change=apply_select_all)

        for m in base_models:
            st.checkbox(m, key=f"chk_{m}")

    run_models = [m for m in base_models if st.session_state.get(f"chk_{m}", False)]
    if not run_models:
        st.info("Select at least one model in the sidebar.")
        return

    if use_sample:
        if not os.path.exists(SAMPLE_TEST_PATH):
            st.warning("Sample test data not found. Please upload a CSV.")
            return
        df = pd.read_csv(SAMPLE_TEST_PATH)
    else:
        if uploaded is None:
            st.info("Upload a test CSV to proceed.")
            return
        df = pd.read_csv(uploaded)

    if not feature_names:
        st.error("Feature metadata not found. Train the models first.")
        return

    try:
        X, y = validate_and_split(df, feature_names, target_name)
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.subheader("Data Preview")
    st.dataframe(df.head())
    st.write(f"Rows: {len(df)}, Features used: {len(feature_names)}")

    def run_for_model(model_name: str) -> Tuple[Dict[str, float] | None, np.ndarray | None, np.ndarray | None]:
        model = load_model(model_name)
        y_pred = model.predict(X)
        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X)
            except Exception:
                y_proba = None
        model_metrics = None
        if y is not None:
            model_metrics = compute_metrics(y, y_pred, y_proba)
        return model_metrics, y_pred, y_proba

    model_outputs: Dict[str, Dict] = {}
    for model_name in run_models:
        with st.spinner(f"Running {model_name}..."):
            mtr, y_pred, y_proba = run_for_model(model_name)
            y_pred_disp = map_array_with_dict(y_pred, idx_to_class) if idx_to_class is not None else y_pred
            model_outputs[model_name] = {
                "metrics": mtr,
                "y_pred": y_pred,
                "y_pred_disp": y_pred_disp,
                "y_proba": y_proba,
            }

    st.subheader("Comparison Table")
    results = []

    for name, out in model_outputs.items():
        if out["metrics"] is not None:
            row = {"ML Model Name": name}
            row.update(out["metrics"])
            results.append(row)
    if results:
        out_df = pd.DataFrame(results)[["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
        out_df = out_df.sort_values(by="Accuracy", ascending=False)
        st.dataframe(out_df, use_container_width=True)

        st.download_button(
            "Download metrics CSV",
            out_df.to_csv(index=False).encode("utf-8"),
            file_name="comparison_metrics.csv"
        )
        st.markdown("Metric comparison")

        metrics_list = ["Accuracy", "F1", "AUC", "Precision", "Recall", "MCC"]
        for row_start in range(0, len(metrics_list), 3):
            row_metrics = metrics_list[row_start:row_start + 3]
            chart_cols = st.columns(3)
            for i, metric in enumerate(row_metrics):
                fig = px.bar(
                    out_df,
                    x="ML Model Name",
                    y=metric,
                    color=metric,
                    color_continuous_scale="RdYlGn",
                    text=out_df[metric].round(3),
                    hover_data={"ML Model Name": True, metric: ":.4f"},
                )
                fig.update_traces(textposition="outside", cliponaxis=False)
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=10, t=20, b=60),
                    yaxis=dict(
                        range=[0, 1.02],
                        title=dict(text=metric, font=dict(size=14)),
                        tickfont=dict(size=12),
                    ),
                    xaxis=dict(
                        title=dict(text="", font=dict(size=14)),
                        tickangle=-35,
                        tickfont=dict(size=12),
                    ),
                    coloraxis_showscale=False,
                    showlegend=False,
                )
                chart_cols[i].plotly_chart(fig, use_container_width=True)

    if y is not None and results:
        st.subheader("Confusion Matrices")
        cols = st.columns(2)
        for i, (name, out) in enumerate(model_outputs.items()):
            if out["metrics"] is None:
                continue

            y_cm, p_cm, labels = normalize_for_confusion(y, out["y_pred_disp"])
            cm = confusion_matrix(y_cm, p_cm, labels=labels)

            fig = plotly_confusion_matrix(cm, labels, title=name)
            cols[i % 2].plotly_chart(fig, use_container_width=False)

    if y is not None and results:
        st.subheader("Classification Reports")
        tabs = st.tabs(list(model_outputs.keys()))
        for tab, (name, out) in zip(tabs, model_outputs.items()):
            if out["metrics"] is None:
                continue
            with tab:
                class_df, summary_df, accuracy_df = classification_report_tables(y, out["y_pred_disp"])
                st.markdown("Per-class metrics")
                try:
                    st.dataframe(class_df, use_container_width=True, hide_index=True)
                except TypeError:
                    st.dataframe(class_df, use_container_width=True)

                st.markdown("Averages")
                try:
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                except TypeError:
                    st.dataframe(summary_df, use_container_width=True)

                acc_val = float(accuracy_df["Accuracy"].iloc[0]) if not accuracy_df.empty else float("nan")
                acc_text = f"{acc_val:.4f}" if not np.isnan(acc_val) else "NA"
                st.markdown(f"**Accuracy: {acc_text}**")

    st.subheader("Predictions")
    if y is not None and results:
        summary_rows = []
        for name, out in model_outputs.items():
            preds = out["y_pred_disp"]
            correct = int(np.sum(y.values == preds))
            total = len(y)
            summary_rows.append({"Model": name, "Outcome": "Correct", "Count": correct})
            summary_rows.append({"Model": name, "Outcome": "Incorrect", "Count": total - correct})
        summary_df = pd.DataFrame(summary_rows)

        st.markdown("Correct vs Incorrect by Model")
        fig1 = px.bar(
            summary_df,
            x="Model",
            y="Count",
            color="Outcome",
            barmode="group",
            text="Count",
            hover_data={"Model": True, "Outcome": True, "Count": True},
        )
        fig1.update_traces(textposition="outside", cliponaxis=False)
        fig1.update_layout(
            height=320,
            margin=dict(l=20, r=10, t=20, b=70),
            xaxis=dict(
                title=dict(text="", font=dict(size=14)),
                tickangle=0,
                tickfont=dict(size=14),
            ),
            yaxis=dict(
                title=dict(text="Count", font=dict(size=14)),
                tickfont=dict(size=14),
            ),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.25,
                xanchor="center",
                x=0.5,
                font=dict(size=14),
            ),
            dragmode="pan",
        )
        st.plotly_chart(fig1, use_container_width=True)

        labels_sorted_num = sorted(pd.unique(y))
        per_class_rows = []
        for name, out in model_outputs.items():
            preds = out["y_pred_disp"]
            for cls in labels_sorted_num:
                mask = (y.values == cls)
                total_c = int(np.sum(mask))
                acc_c = float(np.sum(preds[mask] == cls) / total_c) if total_c > 0 else np.nan
                per_class_rows.append({"Class": str(cls), "Model": name, "Per-class accuracy": acc_c})
        per_class_df = pd.DataFrame(per_class_rows).dropna()

        st.markdown("Per-class Accuracy by Model")
        fig2 = px.bar(
            per_class_df,
            x="Class",
            y="Per-class accuracy",
            color="Model",
            barmode="group",
            text=per_class_df["Per-class accuracy"].round(3),
            hover_data={"Class": True, "Model": True, "Per-class accuracy": ":.4f"},
        )
        fig2.update_traces(textposition="outside", cliponaxis=False)
        fig2.update_layout(
            height=320,
            margin=dict(l=20, r=10, t=20, b=90),
            yaxis=dict(
                range=[0, 1.02],
                title=dict(text="Accuracy", font=dict(size=14)),
                tickfont=dict(size=14),
            ),
            xaxis=dict(
                title=dict(text="", font=dict(size=14)),
                tickfont=dict(size=14),
            ),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.35,
                xanchor="center",
                x=0.5,
                font=dict(size=14),
            ),
            dragmode="pan",
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Top Misclassifications")
        tabs = st.tabs(list(model_outputs.keys()))
        for tab, (name, out) in zip(tabs, model_outputs.items()):
            with tab:
                y_cm, p_cm, labels = normalize_for_confusion(y, out["y_pred_disp"])
                cm = confusion_matrix(y_cm, p_cm, labels=labels)

                cm_no_diag = cm.copy()
                np.fill_diagonal(cm_no_diag, 0)

                err_pairs = []
                for i, a in enumerate(labels):
                    for j, p in enumerate(labels):
                        c = int(cm_no_diag[i, j])
                        if c > 0:
                            err_pairs.append({"Actual": a, "Predicted": p, "Count": c})

                if err_pairs:
                    def actual_sort_key(val: str):
                        try:
                            return (0, int(val))
                        except Exception:
                            return (1, val)

                    err_df = pd.DataFrame(err_pairs)
                    err_df["ActualSort"] = err_df["Actual"].map(actual_sort_key)
                    err_df = err_df.sort_values(by=["ActualSort", "Count"], ascending=[True, False])
                    err_df = err_df.drop(columns=["ActualSort"]).head(5).reset_index(drop=True)

                    st.dataframe(err_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No misclassifications.")

    with st.expander("Preview predictions and download"):
        cols = st.columns(2)
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, (name, out) in enumerate(model_outputs.items()):
                if y is not None:
                    table_df = pd.DataFrame({"Actual": y.values, "Predicted": out["y_pred_disp"]}).reset_index(drop=True)
                else:
                    table_df = pd.DataFrame({"Predicted": out["y_pred_disp"]}).reset_index(drop=True)

                with cols[i % 2]:
                    st.markdown(f"##### {name}")
                    try:
                        st.dataframe(table_df, use_container_width=True, hide_index=True)
                    except TypeError:
                        st.dataframe(table_df, use_container_width=True)

                    st.download_button(
                        f"Download predictions – {name}",
                        table_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"predictions_{name}.csv",
                        key=f"dl_pred_{name}"
                    )
                zf.writestr(f"predictions_{name}.csv", table_df.to_csv(index=False))
        st.download_button(
            "Download ALL predictions (ZIP)",
            data=zip_buf.getvalue(),
            file_name="predictions_all_models.zip",
            mime="application/zip"
        )

    st.markdown("---")
    st.caption("Developed by:\nAkshaya Raj S A")

if __name__ == "__main__":
    main()
