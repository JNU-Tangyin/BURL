import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df1 = pd.read_excel('Baseline.xlsx', sheet_name='RF_Precision')
df2 = pd.read_excel('Baseline.xlsx', sheet_name='RF_Recall')
df3 = pd.read_excel('Baseline.xlsx', sheet_name='RF_F1')
df4 = pd.read_excel('Baseline.xlsx', sheet_name='RF_Accuracy')

fig, axs = plt.subplots(1, 4, figsize=(24, 6.5), subplot_kw=dict(polar=True))

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def plot_radar_chart(ax, df, title):
    df_transposed = df.set_index('Dataset').T.reset_index()
    df_transposed.rename(columns={'index': 'Models'}, inplace=True)
    categories = list(df_transposed)[1:]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    colors = ["#E64B35CC", "#4DBBD5CC", "#00A087CC", "#3C5488CC", "#F39B7FCC", "#8491B4CC", "#91D1c2CC", "#808080",
              "#7E6148CC"]

    df_normalized = normalize(df_transposed.iloc[:, 1:])
    df_normalized['Models'] = df_transposed['Models']

    ax.set_title(title, size=30, color='black', position=(0.5, 1.1))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color='black', size=20)

    for i, color in zip(range(len(df_transposed)), colors):
        data = df_normalized.iloc[i, :-1].values.flatten().tolist()
        data += data[:1]
        ax.plot(angles, data, linewidth=4, linestyle='solid', label=df_transposed.iloc[i]['Models'], color=color)
        ax.fill(angles, data, color=color, alpha=0.25)


plot_radar_chart(axs[0], df1, 'Precision')
plot_radar_chart(axs[1], df2, 'Recall')
plot_radar_chart(axs[2], df3, 'F1 Score')
plot_radar_chart(axs[3], df4, 'Accuracy')

plt.rcParams.update({'font.size': 20})
fig.legend(axs[0].get_legend_handles_labels()[0], axs[0].get_legend_handles_labels()[1], loc='lower center', ncol=9,
           bbox_to_anchor=(0.5, -0.015))

# 自动调整子图布局
plt.tight_layout()
plt.savefig("Comparison.pdf")
plt.show()