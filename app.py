"""
Credit Card Fraud Detection — Flask Web App
==========================================
Run:
    pip install flask pandas numpy scikit-learn matplotlib seaborn
    python app.py creditcard.csv
Then open: http://localhost:5000
"""

import io, base64, warnings, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, request, jsonify, render_template, make_response
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, cohen_kappa_score, log_loss,
    confusion_matrix, roc_curve,
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

app = Flask(__name__)
STATE = {}


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def train_models(csv_path="creditcard.csv"):
    print("Loading dataset ...")
    df = pd.read_csv(csv_path)

    # Separate scalers for Amount and Time
    amount_scaler = StandardScaler()
    time_scaler   = StandardScaler()
    df["Amount_scaled"] = amount_scaler.fit_transform(df[["Amount"]])
    df["Time_scaled"]   = time_scaler.fit_transform(df[["Time"]])
    df_proc = df.drop(columns=["Amount", "Time"])

    fraud_df   = df_proc[df_proc["Class"] == 1]
    legit_df   = df_proc[df_proc["Class"] == 0]
    legit_samp = legit_df.sample(n=len(fraud_df), random_state=42)
    balanced   = pd.concat([fraud_df, legit_samp]).sample(frac=1, random_state=42)

    X = balanced.drop(columns=["Class"])
    y = balanced["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training models ...")
    lr = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
    rf = RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=5,
                                    learning_rate=0.1, subsample=0.8,
                                    random_state=42)
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    models = {
        "Logistic Regression": (lr, "#3498db"),
        "Random Forest":       (rf, "#e67e22"),
        "Gradient Boosting":   (gb, "#9b59b6"),
    }

    def metrics(name, model):
        pred  = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        n     = len(y_test)
        acc   = accuracy_score(y_test, pred)
        return {
            "Model":          name,
            "Accuracy":       round(acc * 100, 2),
            "Precision":      round(precision_score(y_test, pred) * 100, 2),
            "Recall":         round(recall_score(y_test, pred) * 100, 2),
            "F1 Score":       round(f1_score(y_test, pred) * 100, 2),
            "AUC-ROC":        round(roc_auc_score(y_test, proba), 4),
            "MCC":            round(matthews_corrcoef(y_test, pred), 4),
            "Cohen's Kappa":  round(cohen_kappa_score(y_test, pred), 4),
            "Log Loss":       round(log_loss(y_test, proba), 4),
            "Z-Score":        round((acc - 0.5) / np.sqrt(0.25 / n), 4),
            "_pred":          pred,
            "_proba":         proba,
        }

    rows = [metrics(n, m) for n, (m, _) in models.items()]
    mdf  = pd.DataFrame(rows).set_index("Model")

    norm = mdf[["F1 Score", "AUC-ROC", "Recall", "MCC", "Precision"]].copy()
    norm["F1 Score"]  /= 100
    norm["Recall"]    /= 100
    norm["Precision"] /= 100
    weights = {"F1 Score": 0.30, "AUC-ROC": 0.25, "Recall": 0.20,
               "MCC": 0.15, "Precision": 0.10}
    composite = sum(norm[c] * w for c, w in weights.items())
    best_name = composite.idxmax()

    # Store real rows from dataset for the sample endpoint
    real_fraud = df[df["Class"] == 1].reset_index(drop=True)
    real_legit = df[df["Class"] == 0].reset_index(drop=True)

    STATE.update({
        "df": df, "df_proc": df_proc, "fraud_df": fraud_df,
        "X": X, "y": y, "X_test": X_test, "y_test": y_test,
        "models": models, "mdf": mdf, "best_name": best_name,
        "amount_scaler": amount_scaler,
        "time_scaler":   time_scaler,
        "feature_cols":  list(X.columns),
        "real_fraud":    real_fraud,
        "real_legit":    real_legit,
    })
    print(f"Done. Best model: {best_name}")


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    resp = make_response(render_template("index.html"))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"]  = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/api/overview")
def api_overview():
    df = STATE["df"]
    return jsonify({
        "total":      int(df.shape[0]),
        "features":   int(df.shape[1]),
        "fraud":      int(df["Class"].sum()),
        "legitimate": int((df["Class"] == 0).sum()),
        "fraud_pct":  round(df["Class"].mean() * 100, 2),
        "best_model": STATE["best_name"],
    })


@app.route("/api/metrics")
def api_metrics():
    mdf = STATE["mdf"]
    display_cols = ["Accuracy", "Precision", "Recall", "F1 Score",
                    "AUC-ROC", "MCC", "Cohen's Kappa", "Log Loss", "Z-Score"]
    out = []
    for name, row in mdf.iterrows():
        entry = {"model": name}
        for c in display_cols:
            entry[c] = row[c]
        out.append(entry)
    return jsonify({"metrics": out, "best": STATE["best_name"]})


@app.route("/api/sample")
def api_sample():
    """
    Returns a REAL row from the loaded dataset.
    Query param ?type=fraud | legit | random (default: random)
    The frontend fills the form with these values — guaranteed real data.
    """
    kind = request.args.get("type", "random").lower()

    if kind == "fraud":
        pool = STATE["real_fraud"]
    elif kind == "legit":
        pool = STATE["real_legit"]
    else:
        pool = STATE["real_fraud"] if random.randint(0, 1) == 0 else STATE["real_legit"]

    row   = pool.sample(n=1).iloc[0]
    label = "fraud" if int(row["Class"]) == 1 else "legit"

    out = {
        "type":   label,
        "amount": round(float(row["Amount"]), 2),
        "time":   int(row["Time"]),
    }
    for i in range(1, 29):
        out[f"V{i}"] = round(float(row[f"V{i}"]), 6)

    return jsonify(out)


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Accepts JSON: { "amount": float, "time": float, "V1"..V28: float }
    Frontend sends uppercase V1..V28. Returns predictions from all 3 models.
    """
    data = request.get_json(force=True)

    try:
        amount = float(data.get("amount", 0))
        time   = float(data.get("time",   0))
        # Accept V1 (uppercase) — that's what frontend now sends
        v_vals = {}
        for i in range(1, 29):
            v_vals[f"V{i}"] = float(data.get(f"V{i}", data.get(f"v{i}", 0)))
    except (TypeError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

    amount_scaled = float(STATE["amount_scaler"].transform([[amount]])[0][0])
    time_scaled   = float(STATE["time_scaler"].transform([[time]])[0][0])

    # Build feature vector in exact training column order (V1..V28, Amount_scaled, Time_scaled)
    feature_cols = STATE["feature_cols"]
    feat_map = {**v_vals,
                "Amount_scaled": amount_scaled,
                "Time_scaled":   time_scaled}

    X_input = np.array([[feat_map.get(c, 0.0) for c in feature_cols]])

    results = []
    for name, (model, color) in STATE["models"].items():
        pred  = int(model.predict(X_input)[0])
        proba = float(model.predict_proba(X_input)[0][1])
        results.append({
            "model":     name,
            "label":     "FRAUD" if pred == 1 else "Legitimate",
            "fraud_pct": round(proba * 100, 2),
            "color":     color,
        })

    return jsonify({"predictions": results})


@app.route("/api/chart/class_dist")
def chart_class_dist():
    df     = STATE["df"]
    counts = df["Class"].value_counts()
    labels = ["Legitimate", "Fraud"]
    colors = ["#2ecc71", "#e74c3c"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(labels, [counts[0], counts[1]], color=colors,
                edgecolor="black", width=0.45)
    axes[0].set_title("Transaction Class Distribution", fontweight="bold")
    axes[0].set_ylabel("Count")
    for i, v in enumerate([counts[0], counts[1]]):
        axes[0].text(i, v + 1000, f"{v:,}", ha="center", fontweight="bold")
    axes[1].pie([counts[0], counts[1]], labels=labels,
                autopct="%1.2f%%", colors=colors, startangle=90,
                explode=(0, 0.1))
    axes[1].set_title("Fraud vs Legitimate (%)", fontweight="bold")
    fig.suptitle("Class Imbalance in Dataset", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return jsonify({"img": fig_to_b64(fig)})


@app.route("/api/chart/amount_dist")
def chart_amount_dist():
    df    = STATE["df"]
    fraud = df[df["Class"] == 1]["Amount"]
    legit = df[df["Class"] == 0]["Amount"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(legit, bins=50, color="#2ecc71", alpha=0.8, edgecolor="black")
    axes[0].set_title("Legitimate Transaction Amounts", fontweight="bold")
    axes[0].set_xlabel("Amount ($)")
    axes[0].set_ylabel("Frequency")
    axes[1].hist(fraud, bins=50, color="#e74c3c", alpha=0.8, edgecolor="black")
    axes[1].set_title("Fraudulent Transaction Amounts", fontweight="bold")
    axes[1].set_xlabel("Amount ($)")
    fig.suptitle(f"Avg Legit: ${legit.mean():.2f}  |  Avg Fraud: ${fraud.mean():.2f}",
                 fontsize=12)
    fig.tight_layout()
    return jsonify({"img": fig_to_b64(fig)})


@app.route("/api/chart/model_compare")
def chart_model_compare():
    mdf    = STATE["mdf"]
    cols   = ["Accuracy", "Precision", "Recall", "F1 Score"]
    colors = ["#3498db", "#e67e22", "#9b59b6"]
    x      = np.arange(len(cols))
    w      = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (name, row) in enumerate(mdf.iterrows()):
        vals = [row[c] for c in cols]
        bars = ax.bar(x + i * w, vals, w, label=name,
                      color=colors[i], edgecolor="black", alpha=0.9)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() - 1.8, f"{bar.get_height():.1f}",
                    ha="center", va="top", fontsize=8, fontweight="bold", color="white")
    ax.set_xticks(x + w)
    ax.set_xticklabels(cols, fontsize=11)
    ax.set_ylim(70, 102)
    ax.set_ylabel("Score (%)")
    ax.set_title("Model Comparison — Accuracy / Precision / Recall / F1", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return jsonify({"img": fig_to_b64(fig)})


@app.route("/api/chart/confusion")
def chart_confusion():
    mdf    = STATE["mdf"]
    models = STATE["models"]
    cmaps  = ["Blues", "Oranges", "Purples"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (name, _), cmap in zip(axes, models.items(), cmaps):
        pred = mdf.loc[name, "_pred"]
        cm   = confusion_matrix(STATE["y_test"], pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax,
                    xticklabels=["Legit", "Fraud"],
                    yticklabels=["Legit", "Fraud"],
                    linewidths=1, linecolor="black")
        ax.set_title(f"{name}\nConfusion Matrix", fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return jsonify({"img": fig_to_b64(fig)})


@app.route("/api/chart/roc")
def chart_roc():
    mdf    = STATE["mdf"]
    y_test = STATE["y_test"]
    colors = ["#3498db", "#e67e22", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for (name, _), color in zip(STATE["models"].items(), colors):
        proba       = mdf.loc[name, "_proba"]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc         = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})", color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison", fontweight="bold")
    ax.legend(fontsize=10)
    fig.tight_layout()
    return jsonify({"img": fig_to_b64(fig)})


@app.route("/api/chart/feature_importance")
def chart_feature_importance():
    X      = STATE["X"]
    models = STATE["models"]
    pairs  = [(n, m) for n, (m, _) in models.items()
              if hasattr(m, "feature_importances_")]

    fig, axes = plt.subplots(1, len(pairs), figsize=(13, 5))
    if len(pairs) == 1:
        axes = [axes]
    colors = ["#e67e22", "#9b59b6"]
    for ax, (name, model), color in zip(axes, pairs, colors):
        imp = pd.Series(model.feature_importances_, index=X.columns)
        top = imp.nlargest(10).sort_values()
        top.plot(kind="barh", ax=ax, color=color, edgecolor="black", alpha=0.9)
        ax.set_title(f"Top 10 Features\n{name}", fontweight="bold")
        ax.set_xlabel("Importance Score")
    fig.suptitle("Feature Importance Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return jsonify({"img": fig_to_b64(fig)})


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "creditcard.csv"
    train_models(csv)
    print("\n  Open http://localhost:5000 in your browser\n")
    app.run(debug=False, port=5000)
