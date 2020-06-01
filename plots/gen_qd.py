import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

data = pd.read_csv('../large_logs/KL/summary.csv')

y_label = "QD-Score"

plt.figure(figsize = (8,5))
# Plot the responses for different events and regions
sns_plot = sns.lineplot(x="Evaluations", y=y_label,
             hue="Algorithm", data=data)

plt.xticks([0, 5000, 10000], fontsize=20)
#plt.yticks([0, 25], fontsize=20) #8Binary
plt.yticks([0, 600], fontsize=20)  #KL
#plt.yticks([0, 500, 1000], fontsize=20)  #MarioGAN

plt.xlabel("Evaluations",fontsize=20)
plt.ylabel(y_label,fontsize=20)

plt.legend(loc='best', prop={'size': 10})
plt.tight_layout()
plt.show()
sns_plot.figure.savefig("KL.pdf")
