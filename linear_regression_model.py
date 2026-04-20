"""
Linear Regression Model for Movie Success Prediction
Predicts: (1) Rotten Tomatoes score  (2) Profit (revenue - budget)
""" 

import ast
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")



DATA_PATH     = "ML_dataset.csv"   # ← point this at your file
TARGET_RT     = "tomatometer_rating"
TARGET_PROFIT = "profit"              # derived below


def parse_list_column(series):
    def _parse(val):
        if isinstance(val, list):
            return val
        try:
            return ast.literal_eval(val)
        except Exception:
            return []
    return series.apply(_parse)


def load_and_clean(path):
    df = pd.read_csv(path)

    df = df[df["year"] >= 2000].reset_index(drop=True) #movies after 2000s

    df["genres"] = parse_list_column(df["genres"])
    df["cast"]   = parse_list_column(df["cast"])

    # coerce numerics
    for col in ["budget", "revenue", "runtime", "tomatometer_rating", "audience_rating"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 0 budget/revenue almost certainly means "unknown", not literally zero
    for col in ["budget", "revenue"]:
        df[col] = df[col].replace(0, np.nan)

    # fill missing numerics with median
    for col in ["budget", "revenue", "runtime", "tomatometer_rating", "audience_rating"]:
        df[col] = df[col].fillna(df[col].median())

    df["profit"] = df["revenue"] - df["budget"]

    # fill missing categorical values 
    df["director"] = df["director"].fillna("Unknown")
    df["year"]     = df["year"].fillna(df["year"].median()).astype(int)

    return df


def engineer_features(df):
    # Multi-hot genres
    mlb = MultiLabelBinarizer()
    genre_dummies = pd.DataFrame(
        mlb.fit_transform(df["genres"]),
        columns=[f"genre_{g}" for g in mlb.classes_],
        index=df.index,
    )

    top_dirs = df["director"].value_counts().nlargest(50).index
    dir_clean = df["director"].where(df["director"].isin(top_dirs), "Other")
    director_dummies = pd.get_dummies(dir_clean, prefix="dir")

    numerical = df[["budget", "runtime", "year"]].copy()

    X = pd.concat([numerical, genre_dummies, director_dummies], axis=1)
    X.columns = X.columns.astype(str)
    return X


def build_pipeline(alpha=1.0):
    return Pipeline([
        ("scaler",    StandardScaler()),
        ("regressor", Ridge(alpha=alpha)),
    ])

def tune_alpha(X_train, y_train, alphas=None):
    if alphas is None:
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 500.0]
    print("  Alpha tuning (5-fold CV RMSE):")
    best_alpha, best_rmse = 1.0, float("inf")
    for a in alphas:
        scores = cross_val_score(build_pipeline(alpha=a), X_train, y_train,
                                 scoring="neg_root_mean_squared_error", cv=5)
        rmse = -scores.mean()
        print(f"    alpha={a:<8}  CV RMSE={rmse:.4f}")
        if rmse < best_rmse:
            best_rmse, best_alpha = rmse, a
    print(f"  → Best alpha: {best_alpha}  (CV RMSE={best_rmse:.4f})\n")
    return best_alpha


def evaluate(pipeline, X, y, split_name="Set"):
    preds = pipeline.predict(X)
    rmse  = np.sqrt(mean_squared_error(y, preds))
    mse   = mean_squared_error(y, preds)
    r2    = r2_score(y, preds)
    print(f"  [{split_name:<6}]  RMSE={rmse:.4f}  |  MSE={mse:.4f}  |  R²={r2:.4f}")
    return {"rmse": rmse, "mse": mse, "r2": r2,
            "preds": preds, "actuals": np.array(y)}


def plot_results(results, target_label):
    actuals, preds = results["actuals"], results["preds"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Linear Regression — {target_label}", fontsize=14)

    # actual vs predicted plot
    ax = axes[0]
    ax.scatter(actuals, preds, alpha=0.45, color="steelblue", edgecolors="white", linewidths=0.3)
    lo, hi = min(actuals.min(), preds.min()), max(actuals.max(), preds.max())
    ax.plot([lo, hi], [lo, hi], "r--", label="Perfect fit")
    ax.set_xlabel(f"Actual {target_label}")
    ax.set_ylabel(f"Predicted {target_label}")
    ax.set_title("Actual vs Predicted (Test Set)")
    ax.legend()

    # residuals
    ax = axes[1]
    residuals = actuals - preds
    ax.scatter(preds, residuals, alpha=0.45, color="coral", edgecolors="white", linewidths=0.3)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel(f"Predicted {target_label}")
    ax.set_ylabel("Residual (Actual − Predicted)")
    ax.set_title("Residual Plot (Test Set)")

    plt.tight_layout()
    fname = f"linreg_{target_label.replace(' ', '_').replace('(','').replace(')','').replace('/','')}.png"
    plt.savefig(fname, dpi=120)
    plt.show()
    print(f"  Plot saved → {fname}\n")


def print_top_coefficients(pipeline, feature_names, n=15):
    coefs = pipeline.named_steps["regressor"].coef_
    coef_df = (pd.DataFrame({"feature": feature_names, "coefficient": coefs})
               .assign(abs=lambda d: d["coefficient"].abs())
               .sort_values("abs", ascending=False)
               .head(n))
    print(f"  Top {n} most influential features (by |coefficient|):")
    print(coef_df[["feature", "coefficient"]].to_string(index=False))
    print()


def run_target(X, y, target_label):
    print(f"\n{'='*58}")
    print(f"  TARGET: {target_label}")
    print(f"{'='*58}")

    # 70 / 15 / 15 split 
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42)

    print(f"  Sizes — train: {len(X_train)}  val: {len(X_val)}  test: {len(X_test)}\n")

    best_alpha = tune_alpha(X_train, y_train)

    pipe = build_pipeline(alpha=best_alpha)
    pipe.fit(X_train, y_train)

    evaluate(pipe, X_train, y_train, "Train")
    evaluate(pipe, X_val,   y_val,   "Val  ")
    test_res = evaluate(pipe, X_test, y_test,  "Test ")

    print_top_coefficients(pipe, list(X.columns))
    plot_results(test_res, target_label)

    return pipe


if __name__ == "__main__":
    df = load_and_clean(DATA_PATH)
    X  = engineer_features(df)

    print(f"Dataset ready — {len(df)} rows, {X.shape[1]} features\n")
    print("Sample feature columns:", list(X.columns[:8]), "...")

    rt_model     = run_target(X, df[TARGET_RT],     "Rotten Tomatoes Score")
    profit_model = run_target(X, df[TARGET_PROFIT],  "Profit (USD)")