"""
Feature Correlation Analysis for BRFSS Diabetes Dataset
Outputs: Correlation with target, inter-feature correlations, and recommendations
"""
import pandas as pd
import numpy as np

CSV = r"C:\Users\ACER\Desktop\ML 2\New Model Training\diabetes_health_indicators_full.csv"
df = pd.read_csv(CSV)

print("=" * 70)
print(" CORRELATION ANALYSIS: BRFSS Diabetes Dataset")
print("=" * 70)

# 1. Correlation with target (Diabetes_binary)
print("\n" + "=" * 70)
print(" 1. FEATURE CORRELATION WITH DIABETES (sorted by strength)")
print("=" * 70)
corr_target = df.corr()["Diabetes_binary"].drop("Diabetes_binary").sort_values(key=abs, ascending=False)
for feat, val in corr_target.items():
    bar = "█" * int(abs(val) * 50)
    direction = "+" if val > 0 else "-"
    strength = "STRONG" if abs(val) > 0.2 else "MODERATE" if abs(val) > 0.1 else "WEAK" if abs(val) > 0.05 else "USELESS"
    print(f"  {feat:<25} {direction}{abs(val):.4f}  {bar}  [{strength}]")

# 2. High inter-feature correlations (potential combinations)
print("\n" + "=" * 70)
print(" 2. HIGHLY CORRELATED FEATURE PAIRS (|r| > 0.3)")
print("    These features could be COMBINED into one")
print("=" * 70)
corr_matrix = df.drop(columns=["Diabetes_binary"]).corr()
pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.3:
            pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], r))
pairs.sort(key=lambda x: abs(x[2]), reverse=True)
for f1, f2, r in pairs:
    print(f"  {f1:<20} ↔ {f2:<20}  r = {r:+.4f}")

# 3. Useless features (very low correlation with target)
print("\n" + "=" * 70)
print(" 3. POTENTIALLY USELESS FEATURES (|r| < 0.05 with Diabetes)")
print("=" * 70)
useless = corr_target[abs(corr_target) < 0.05]
for feat, val in useless.items():
    print(f"  ❌ {feat:<25} r = {val:+.4f}")

# 4. Summary recommendations
print("\n" + "=" * 70)
print(" 4. RECOMMENDATIONS")
print("=" * 70)
print("""
  COMBINE (highly correlated with each other):
    → HighBP + HighChol + HeartDiseaseorAttack → "Cardiovascular Risk"
    → Fruits + Veggies → "Healthy Diet"
    → Age + DiffWalk → "Age-related Fragility"

  CONSIDER REMOVING (very low predictive power):
    → Features with |r| < 0.05 listed above

  KEEP (strong individual predictors):
    → GenHlth, HighBP, Age, BMI, HighChol, DiffWalk, Income
""")
print("=" * 70)
