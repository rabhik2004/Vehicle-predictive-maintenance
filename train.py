"""Train + evaluate LR / DT / RF / XGBoost, run SHAP, produce health reports."""
import json, warnings
import joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from generate_data import generate
from features import add_features, feature_columns, base_sensor
from health import HealthMonitor

warnings.filterwarnings("ignore")
SEED = 42

raw = generate(seed=SEED)
raw.to_csv("vehicle_sensor_data.csv", index=False)
df = add_features(raw).reset_index(drop=True)
feats = feature_columns(df)

# split BY VEHICLE so no vehicle appears in both train and test (avoids leakage)
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
tr, te = next(gss.split(df, df.label, df.vehicle_id))
Xtr, Xte, ytr, yte = df.loc[tr, feats], df.loc[te, feats], df.label[tr], df.label[te]
pos_w = (ytr == 0).sum() / (ytr == 1).sum()

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, class_weight="balanced",
                                            n_jobs=-1, random_state=SEED),
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.9,
                             colsample_bytree=0.8, scale_pos_weight=pos_w, eval_metric="logloss",
                             random_state=SEED, n_jobs=-1),
}

rows, probs = [], {}
for name, m in models.items():
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]; probs[name] = p
    pred = (p >= 0.5).astype(int)
    rows.append(dict(model=name, accuracy=accuracy_score(yte, pred), precision=precision_score(yte, pred),
                     recall=recall_score(yte, pred), f1=f1_score(yte, pred), roc_auc=roc_auc_score(yte, p)))
res = pd.DataFrame(rows).round(4)
res.to_csv("results.csv", index=False)
print(res.to_string(index=False))

# ROC curves
plt.figure(figsize=(6, 5))
for n, p in probs.items():
    fpr, tpr, _ = roc_curve(yte, p); plt.plot(fpr, tpr, label=f"{n} (AUC={roc_auc_score(yte, p):.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=.8); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curves - failure within next 20 readings"); plt.legend(); plt.tight_layout()
plt.savefig("roc_curves.png", dpi=140); plt.close()

# best model by F1 among tree models (SHAP TreeExplainer needs a tree model)
tree_names = ["Decision Tree", "Random Forest", "XGBoost"]
best = max(tree_names, key=lambda n: res.set_index("model").loc[n, "f1"])
model = models[best]; print("Best tree model:", best)
ConfusionMatrixDisplay(confusion_matrix(yte, (probs[best] >= .5).astype(int)),
                       display_labels=["OK", "Failure soon"]).plot(cmap="Blues")
plt.title(f"Confusion matrix - {best}"); plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=140); plt.close()

# global SHAP, aggregated to sensor level
samp = Xte.sample(1500, random_state=SEED)
sv = shap.TreeExplainer(model).shap_values(samp)
sv = sv[1] if isinstance(sv, list) else sv
sv = sv[..., 1] if getattr(sv, "ndim", 2) == 3 else sv
imp = pd.Series(np.abs(sv).mean(0), index=feats)
sensor_imp = imp.groupby(base_sensor).sum().sort_values()
plt.figure(figsize=(7, 4.5)); sensor_imp.plot.barh(color="#2b6cb0")
plt.xlabel("Mean |SHAP value|"); plt.title(f"Sensor importance (SHAP) - {best}"); plt.tight_layout()
plt.savefig("shap_sensor_importance.png", dpi=140); plt.close()
shap.summary_plot(sv, samp, max_display=15, show=False); plt.tight_layout()
plt.savefig("shap_summary.png", dpi=140); plt.close()

# health-monitor demo on 3 test snapshots (healthy / warning / critical)
healthy_ref = raw[(raw.label == 0)]
mon = HealthMonitor(model, feats, healthy_ref)
te_df = df.loc[te].copy(); te_df["p"] = probs[best]
demo = {}
for tag, target in [("healthy", 0.05), ("warning", 0.55), ("critical", 0.95)]:
    i = (te_df.p - target).abs().idxmin()
    demo[tag] = mon.assess(df.loc[[i]], df.loc[i])
    print(f"\n--- {tag.upper()} sample ---\n" + mon.report(demo[tag]))

joblib.dump(dict(model=model, feats=feats, base=mon.base), "model.joblib")
json.dump(dict(best_model=best, demo=demo, results=res.to_dict("records")), open("summary.json", "w"), indent=2)
