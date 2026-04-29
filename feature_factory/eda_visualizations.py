# Visualization 1: Sector Distribution Bar Chart
import matplotlib.pyplot as plt
import numpy as np

sector_counts = dfPlot['sector'].value_counts()
colors = plt.cm.Set2(np.linspace(0, 1, len(sector_counts)))

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(sector_counts.index, sector_counts.values, color=colors, edgecolor='white', linewidth=0.5)
ax.set_xlabel('Number of Stocks', fontsize=12)
ax.set_title('Stock Distribution by Sector', fontsize=14, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, val in zip(bars, sector_counts.values):
    ax.text(val + 15, bar.get_y() + bar.get_height() / 2, f'{val:,}', va='center', fontsize=10)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()


# Visualization 2: Expected Upside Distribution (MC vs Kalman vs PT)
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(12, 6))
bins = np.linspace(-100, 200, 80)
ax.hist(dfPlot['expected_upside_mc'].dropna().clip(-100, 200), bins=bins, alpha=0.5,
        label='Monte Carlo', color='#2A9D8F', edgecolor='white', linewidth=0.3)
ax.hist(dfPlot['expected_upside_kalman'].dropna().clip(-100, 200), bins=bins, alpha=0.5,
        label='Kalman Filter', color='#E76F51', edgecolor='white', linewidth=0.3)
ax.hist(dfPlot['expected_upside_pt'].dropna().clip(-100, 200), bins=bins, alpha=0.5,
        label='Price Target', color='#264653', edgecolor='white', linewidth=0.3)
ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.7, label='Break-even')
ax.set_xlabel('Expected Upside (%)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Expected Upside: Three Model Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.2)
plt.tight_layout()
plt.show()


# Visualization 3: Scatter — Prob Positive Upside vs Expected Upside MC colored by Region
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(12, 7))
regions = dfPlot['region'].unique()
cmap = plt.cm.tab10(np.linspace(0, 1, len(regions)))
for i, region in enumerate(regions):
    mask = dfPlot['region'] == region
    ax.scatter(
        dfPlot.loc[mask, 'prob_positive_upside'],
        dfPlot.loc[mask, 'expected_upside_mc'].clip(-100, 300),
        alpha=0.35, s=12, color=cmap[i], label=region, rasterized=True
    )
ax.axhline(0, color='grey', linestyle='--', linewidth=0.8, alpha=0.6)
ax.set_xlabel('Probability of Positive Upside (%)', fontsize=12)
ax.set_ylabel('Expected Upside MC (%, clipped)', fontsize=12)
ax.set_title('Probability of Positive Upside vs Expected Return by Region', fontsize=14, fontweight='bold')
ax.legend(title='Region', fontsize=9, title_fontsize=10, markerscale=3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()


# Visualization 4: Box Plot — Risk-Reward Ratio by Size Class and Style Class
import matplotlib.pyplot as plt

size_classes = dfPlot['size_class'].unique()
style_classes = dfPlot['style_class'].unique()

fig, axes = plt.subplots(1, len(size_classes), figsize=(16, 6), sharey=True)
fig.suptitle('Risk-Reward Ratio by Size Class & Style Class', fontsize=14, fontweight='bold')
colors = {'Growth': '#2A9D8F', 'Value': '#E76F51', 'Blend': '#264653'}

for idx, sc in enumerate(sorted(size_classes)):
    ax = axes[idx]
    data_groups = []
    labels = []
    for st in sorted(style_classes):
        vals = dfPlot.loc[(dfPlot['size_class'] == sc) & (dfPlot['style_class'] == st), 'risk_reward_ratio']
        vals = vals.dropna().clip(-50, 50)
        data_groups.append(vals.values)
        labels.append(st)
    bp = ax.boxplot(data_groups, labels=labels, patch_artist=True, widths=0.6,
                    medianprops=dict(color='black', linewidth=1.5),
                    flierprops=dict(marker='.', markersize=2, alpha=0.3))
    for patch, label in zip(bp['boxes'], labels):
        patch.set_facecolor(colors.get(label, '#999999'))
        patch.set_alpha(0.7)
    ax.set_title(f'{sc}', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0].set_ylabel('Risk-Reward Ratio (clipped ±50)', fontsize=11)
plt.tight_layout()
plt.show()


# Visualization 5: Heatmap — Correlation Matrix of Key Numeric Features
import matplotlib.pyplot as plt

numeric_cols = [
    'expected_upside_mc', 'expected_upside_kalman', 'expected_upside_pt',
    'implied_return_mc', 'implied_return_kalman', 'implied_return_pt',
    'prob_positive_upside', 'risk_reward_ratio', 'signal_strength',
    'upside_std', 'kalman_gain', 'var_5_pct'
]
corr = dfPlot[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(numeric_cols)))
ax.set_yticks(range(len(numeric_cols)))
short_labels = [c.replace('expected_upside_', 'eu_').replace('implied_return_', 'ir_') for c in numeric_cols]
ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(short_labels, fontsize=9)
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        val = corr.values[i, j]
        color = 'white' if abs(val) > 0.6 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Pearson Correlation')
ax.set_title('Correlation Matrix of Key Return & Risk Metrics', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# Visualization 6: Region vs Sector Bubble Chart (count + mean upside)
import matplotlib.pyplot as plt

grouped = dfPlot.groupby(['region', 'sector']).agg(
    count=('ticker', 'size'),
    mean_upside=('expected_upside_mc', 'mean')
).reset_index()

regions_list = sorted(grouped['region'].unique())
sectors_list = sorted(grouped['sector'].unique())
region_map = {r: i for i, r in enumerate(regions_list)}
sector_map = {s: i for i, s in enumerate(sectors_list)}

fig, ax = plt.subplots(figsize=(14, 7))
sc = ax.scatter(
    grouped['region'].map(region_map),
    grouped['sector'].map(sector_map),
    s=grouped['count'] * 1.5,
    c=grouped['mean_upside'],
    cmap='RdYlGn', edgecolors='grey', linewidth=0.5, alpha=0.85,
    vmin=-20, vmax=80
)
ax.set_xticks(range(len(regions_list)))
ax.set_xticklabels(regions_list, fontsize=10)
ax.set_yticks(range(len(sectors_list)))
ax.set_yticklabels(sectors_list, fontsize=9)
ax.set_xlabel('Region', fontsize=12)
ax.set_ylabel('Sector', fontsize=12)
ax.set_title('Region × Sector: Stock Count (bubble size) & Mean Expected Upside (color)',
             fontsize=13, fontweight='bold')
plt.colorbar(sc, ax=ax, label='Mean Expected Upside MC (%)', fraction=0.03, pad=0.02)
ax.grid(alpha=0.15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()


# Visualization 7: Signal Strength vs Kalman Variance — 2D Hexbin Density
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 7))
hb = ax.hexbin(
    dfPlot['signal_strength'], dfPlot['kalman_variance'],
    gridsize=40, cmap='YlOrRd', mincnt=1, linewidths=0.2
)
ax.set_xlabel('Signal Strength', fontsize=12)
ax.set_ylabel('Kalman Variance', fontsize=12)
ax.set_title('Signal Strength vs Kalman Variance (Hexbin Density)', fontsize=14, fontweight='bold')
plt.colorbar(hb, ax=ax, label='Count', fraction=0.03, pad=0.02)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(alpha=0.15)
plt.tight_layout()
plt.show()


# Visualization 8: Top 15 Countries by Median Expected Upside with IQR Error Bars
import matplotlib.pyplot as plt
import numpy as np

country_stats = dfPlot.groupby('country')['expected_upside_mc'].agg(['median', 'count']).reset_index()
country_stats = country_stats[country_stats['count'] >= 10].sort_values('median', ascending=True).tail(15)

q1 = dfPlot.groupby('country')['expected_upside_mc'].quantile(0.25)
q3 = dfPlot.groupby('country')['expected_upside_mc'].quantile(0.75)
country_stats = country_stats.set_index('country')
country_stats['q1'] = q1
country_stats['q3'] = q3
country_stats = country_stats.dropna()

fig, ax = plt.subplots(figsize=(10, 7))
y_pos = np.arange(len(country_stats))
xerr_low = country_stats['median'] - country_stats['q1']
xerr_high = country_stats['q3'] - country_stats['median']
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(country_stats)))

ax.barh(y_pos, country_stats['median'], xerr=[xerr_low.values, xerr_high.values],
        color=colors, edgecolor='white', linewidth=0.5, capsize=3, error_kw={'linewidth': 1, 'alpha': 0.6})
ax.set_yticks(y_pos)
ax.set_yticklabels(country_stats.index, fontsize=10)
ax.set_xlabel('Median Expected Upside MC (%)', fontsize=12)
ax.set_title('Top 15 Countries by Median Expected Upside (with IQR)', fontsize=14, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', alpha=0.3)
for i, (med, cnt) in enumerate(zip(country_stats['median'], country_stats['count'])):
    ax.text(med + 1, i, f'n={int(cnt)}', va='center', fontsize=8, alpha=0.7)
plt.tight_layout()
plt.show()
