import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("outputs/qini_curve.csv")
plt.figure(figsize=(8, 5))
plt.plot(df["rank"], df["qini"], label="DML Model", color="#1f77b4", lw=2)
plt.plot([0, df["rank"].max()], [0, df["qini"].iloc[-1]], "--", color="gray", label="Random")
plt.title("Qini Curve (Uplift)")
plt.xlabel("Number of Customers Targeted")
plt.ylabel("Cumulative Incremental Conversions")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/qini_plot.png", dpi=300)
