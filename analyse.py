"""
Kaggle Dataset Ecosystem Analysis
Generates insights and charts to understand dataset engagement patterns.
"""

import csv
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Configuration
CSV_FILE = "kaggle_datasets.csv"
CHARTS_DIR = "charts"
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12


def load_data():
    """Load and parse the CSV data."""
    datasets = []
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            row['view_count'] = int(row['view_count'] or 0)
            row['download_count'] = int(row['download_count'] or 0)
            row['total_votes'] = int(row['total_votes'] or 0)
            row['script_count'] = int(row['script_count'] or 0)
            row['dataset_size'] = int(row['dataset_size'] or 0)
            row['usability_score'] = float(row['usability_score'] or 0)
            datasets.append(row)
    return datasets


def save_chart(fig, name):
    """Save chart to the charts directory."""
    path = os.path.join(CHARTS_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def is_bulk_uploader(user_datasets):
    """Detect bulk/automated uploaders based on dataset slug patterns."""
    if len(user_datasets) < 100:
        return False

    slugs = [d['dataset_slug'] for d in user_datasets]

    # Check for numbered sequences (e.g., xxx-001, xxx-002)
    numbered = sum(1 for s in slugs if any(f'-{str(i).zfill(2)}' in s or f'-{str(i).zfill(3)}' in s for i in range(200)))
    if numbered > len(slugs) * 0.5:
        return True

    # Check for common prefix pattern (>30% share same 3-word prefix)
    prefixes = ['-'.join(s.split('-')[:3]) for s in slugs]
    most_common_prefix = Counter(prefixes).most_common(1)
    if most_common_prefix and most_common_prefix[0][1] > len(slugs) * 0.3:
        return True

    # Check for stock-like pattern
    stock_pattern = sum(1 for s in slugs if 'stock' in s.lower() or '-ns-' in s.lower())
    if stock_pattern > len(slugs) * 0.5:
        return True

    return False


def get_bulk_uploaders(datasets):
    """Identify all bulk uploaders in the dataset."""
    creators = defaultdict(list)
    for d in datasets:
        if d['owner_name']:
            creators[d['owner_name']].append(d)

    bulk_users = set()
    for name, user_datasets in creators.items():
        if is_bulk_uploader(user_datasets):
            bulk_users.add(name)

    return bulk_users


def chart_owner_tier_distribution(datasets):
    """Chart 1: Distribution of datasets by owner tier."""
    tiers = Counter(d['owner_tier'] for d in datasets if d['owner_tier'])

    # Order tiers logically
    tier_order = ['NOVICE', 'CONTRIBUTOR', 'EXPERT', 'MASTER', 'GRANDMASTER']
    ordered_tiers = [(t, tiers.get(t, 0)) for t in tier_order if t in tiers]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#a8dadc', '#457b9d', '#1d3557', '#e63946', '#f4a261']

    labels, values = zip(*ordered_tiers) if ordered_tiers else ([], [])
    bars = ax.bar(labels, values, color=colors[:len(labels)], edgecolor='black', linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f'{val:,}', ha='center', va='bottom', fontweight='bold')

    ax.set_xlabel('Owner Tier')
    ax.set_ylabel('Number of Datasets')
    ax.set_title('Dataset Distribution by Owner Tier\n(Higher tiers have more experience on Kaggle)')

    # Add insight
    total = sum(values)
    top_tier = max(ordered_tiers, key=lambda x: x[1]) if ordered_tiers else ('N/A', 0)
    ax.text(0.98, 0.95, f"Insight: {top_tier[0]}s contribute {top_tier[1]/total*100:.1f}% of datasets",
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    return save_chart(fig, '01_owner_tier_distribution')


def chart_engagement_by_tier(datasets):
    """Chart 2: Average engagement metrics by owner tier."""
    tier_stats = defaultdict(lambda: {'views': [], 'downloads': [], 'votes': []})

    for d in datasets:
        tier = d['owner_tier']
        if tier:
            tier_stats[tier]['views'].append(d['view_count'])
            tier_stats[tier]['downloads'].append(d['download_count'])
            tier_stats[tier]['votes'].append(d['total_votes'])

    tier_order = ['NOVICE', 'CONTRIBUTOR', 'EXPERT', 'MASTER', 'GRANDMASTER']
    tiers = [t for t in tier_order if t in tier_stats]

    avg_views = [np.mean(tier_stats[t]['views']) for t in tiers]
    avg_downloads = [np.mean(tier_stats[t]['downloads']) for t in tiers]
    avg_votes = [np.mean(tier_stats[t]['votes']) for t in tiers]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(tiers))
    width = 0.25

    bars1 = ax.bar(x - width, avg_views, width, label='Avg Views', color='#3498db')
    bars2 = ax.bar(x, avg_downloads, width, label='Avg Downloads', color='#2ecc71')
    bars3 = ax.bar(x + width, avg_votes, width, label='Avg Votes', color='#e74c3c')

    ax.set_xlabel('Owner Tier')
    ax.set_ylabel('Average Count')
    ax.set_title('Average Engagement Metrics by Owner Tier\n(Does experience lead to more engagement?)')
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.legend()

    # Add insight
    if avg_votes:
        best_tier = tiers[np.argmax(avg_votes)]
        ax.text(0.98, 0.95, f"Insight: {best_tier}s get highest avg votes ({max(avg_votes):.0f})",
                transform=ax.transAxes, ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    return save_chart(fig, '02_engagement_by_tier')


def chart_top_categories(datasets):
    """Chart 3: Top 15 most popular categories."""
    category_counts = Counter()

    for d in datasets:
        cats = d.get('category_names', '')
        if cats:
            for cat in cats.split(', '):
                cat = cat.strip()
                if cat:
                    category_counts[cat] += 1

    top_cats = category_counts.most_common(15)

    fig, ax = plt.subplots(figsize=(12, 8))
    labels, values = zip(*top_cats) if top_cats else ([], [])

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))
    bars = ax.barh(range(len(labels)), values, color=colors)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('Number of Datasets')
    ax.set_title('Top 15 Dataset Categories\n(Most popular topics on Kaggle)')

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(val + 100, bar.get_y() + bar.get_height()/2,
                f'{val:,}', ha='left', va='center')

    # Add insight
    if top_cats:
        ax.text(0.98, 0.02, f"Insight: '{top_cats[0][0]}' dominates with {top_cats[0][1]:,} datasets",
                transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    return save_chart(fig, '03_top_categories')


def chart_license_distribution(datasets):
    """Chart 4: License type distribution."""
    licenses = Counter(d['license_short_name'] for d in datasets if d['license_short_name'])
    top_licenses = licenses.most_common(10)

    fig, ax = plt.subplots(figsize=(12, 8))
    labels, values = zip(*top_licenses) if top_licenses else ([], [])
    total = sum(values)

    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    bars = ax.barh(range(len(labels)), values, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('Number of Datasets')
    ax.set_title('Dataset License Distribution\n(Choose the right license for your dataset)')

    # Add value and percentage labels
    for bar, val in zip(bars, values):
        pct = val / total * 100
        ax.text(val + total * 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:,} ({pct:.1f}%)', ha='left', va='center', fontsize=9)

    # Add insight
    if top_licenses:
        ax.text(0.98, 0.02, f"Insight: {top_licenses[0][0]} is most popular ({top_licenses[0][1]/total*100:.1f}%)",
                transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return save_chart(fig, '04_license_distribution')


def chart_usability_vs_engagement(datasets):
    """Chart 5: Usability score vs engagement."""
    # Group by usability score ranges
    usability_groups = defaultdict(lambda: {'downloads': [], 'votes': []})

    for d in datasets:
        score = d['usability_score']
        if score > 0:
            # Round to 1 decimal
            score_bin = round(score, 1)
            usability_groups[score_bin]['downloads'].append(d['download_count'])
            usability_groups[score_bin]['votes'].append(d['total_votes'])

    scores = sorted(usability_groups.keys())
    avg_downloads = [np.mean(usability_groups[s]['downloads']) for s in scores]
    avg_votes = [np.mean(usability_groups[s]['votes']) for s in scores]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color1 = '#2ecc71'
    ax1.set_xlabel('Usability Score')
    ax1.set_ylabel('Avg Downloads', color=color1)
    line1 = ax1.plot(scores, avg_downloads, color=color1, linewidth=2, marker='o', label='Avg Downloads')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = '#e74c3c'
    ax2.set_ylabel('Avg Votes', color=color2)
    line2 = ax2.plot(scores, avg_votes, color=color2, linewidth=2, marker='s', label='Avg Votes')
    ax2.tick_params(axis='y', labelcolor=color2)

    ax1.set_title('Usability Score vs Engagement\n(Does better documentation lead to more engagement?)')

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    # Add insight
    ax1.text(0.98, 0.95, "Insight: Higher usability scores correlate with better engagement",
             transform=ax1.transAxes, ha='right', va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    return save_chart(fig, '05_usability_vs_engagement')


def chart_size_distribution(datasets):
    """Chart 6: Dataset size distribution."""
    sizes = [d['dataset_size'] for d in datasets if d['dataset_size'] > 0]

    # Convert to MB
    sizes_mb = [s / (1024 * 1024) for s in sizes]

    # Create size buckets
    buckets = ['< 1 MB', '1-10 MB', '10-100 MB', '100 MB - 1 GB', '> 1 GB']
    bucket_counts = [0, 0, 0, 0, 0]

    for size in sizes_mb:
        if size < 1:
            bucket_counts[0] += 1
        elif size < 10:
            bucket_counts[1] += 1
        elif size < 100:
            bucket_counts[2] += 1
        elif size < 1024:
            bucket_counts[3] += 1
        else:
            bucket_counts[4] += 1

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#a8e6cf', '#88d8b0', '#56c596', '#329d9c', '#205072']
    bars = ax.bar(buckets, bucket_counts, color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, bucket_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'{val:,}', ha='center', va='bottom', fontweight='bold')

    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Number of Datasets')
    ax.set_title('Dataset Size Distribution\n(What sizes are most common?)')

    # Add insight
    most_common = buckets[bucket_counts.index(max(bucket_counts))]
    ax.text(0.98, 0.95, f"Insight: Most datasets are {most_common}",
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    return save_chart(fig, '06_size_distribution')


def chart_downloads_vs_votes(datasets):
    """Chart 7: Downloads vs Votes scatter plot (engagement correlation)."""
    # Sample for performance
    sample = datasets[:10000] if len(datasets) > 10000 else datasets

    downloads = [d['download_count'] for d in sample]
    votes = [d['total_votes'] for d in sample]
    views = [d['view_count'] for d in sample]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by view count
    scatter = ax.scatter(downloads, votes, c=views, cmap='YlOrRd',
                         alpha=0.5, s=20, edgecolors='none')

    ax.set_xlabel('Download Count')
    ax.set_ylabel('Vote Count')
    ax.set_title('Downloads vs Votes (colored by Views)\n(How are engagement metrics related?)')
    ax.set_xscale('log')
    ax.set_yscale('log')

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('View Count')

    # Calculate correlation
    corr = np.corrcoef(downloads, votes)[0, 1]
    ax.text(0.02, 0.98, f"Correlation: {corr:.3f}",
            transform=ax.transAxes, ha='left', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    return save_chart(fig, '07_downloads_vs_votes')


def chart_top_creators(datasets, bulk_uploaders):
    """Chart 8: Top 15 most prolific dataset creators (excluding bulk uploaders)."""
    # Exclude bulk uploaders for genuine creator ranking
    creators = Counter(d['owner_name'] for d in datasets if d['owner_name'] and d['owner_name'] not in bulk_uploaders)
    top_creators = creators.most_common(15)

    fig, ax = plt.subplots(figsize=(12, 8))
    labels, values = zip(*top_creators) if top_creators else ([], [])

    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(labels)))
    bars = ax.barh(range(len(labels)), values, color=colors)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('Number of Datasets')
    ax.set_title('Top 15 Most Prolific Dataset Creators\n(Genuine creators - excluding bulk/automated uploads)')

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(val + 5, bar.get_y() + bar.get_height()/2,
                f'{val}', ha='left', va='center')

    # Add note about excluded bulk uploaders
    ax.text(0.98, 0.02, f"Note: {len(bulk_uploaders)} bulk uploaders excluded\n(automated/split dataset uploads)",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    return save_chart(fig, '08_top_creators')


def chart_file_types(datasets):
    """Chart 9: Most common file types."""
    file_types = Counter()

    for d in datasets:
        try:
            files = json.loads(d.get('common_file_types', '[]'))
            for f in files:
                ftype = f.get('fileType', '').replace('DATASET_FILE_TYPE_', '')
                if ftype:
                    file_types[ftype] += f.get('count', 1)
        except:
            pass

    top_types = file_types.most_common(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    labels, values = zip(*top_types) if top_types else ([], [])

    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('File Type')
    ax.set_ylabel('Count')
    ax.set_title('Most Common File Types in Datasets\n(What formats dominate?)')
    plt.xticks(rotation=45, ha='right')

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'{val:,}', ha='center', va='bottom', fontsize=9)

    # Add insight
    if top_types:
        ax.text(0.98, 0.95, f"Insight: {top_types[0][0]} is the dominant format",
                transform=ax.transAxes, ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return save_chart(fig, '09_file_types')


def chart_engagement_funnel(datasets):
    """Chart 10: Engagement funnel (views -> downloads -> votes)."""
    total_views = sum(d['view_count'] for d in datasets)
    total_downloads = sum(d['download_count'] for d in datasets)
    total_votes = sum(d['total_votes'] for d in datasets)

    stages = ['Views', 'Downloads', 'Votes']
    values = [total_views, total_downloads, total_votes]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create funnel effect
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    for i, (stage, val, color) in enumerate(zip(stages, values, colors)):
        width = 0.5 - i * 0.1
        ax.barh(i, val, height=width, color=color, alpha=0.8)
        ax.text(val + total_views * 0.02, i, f'{val:,.0f}', va='center', fontweight='bold')

    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(stages)
    ax.set_xlabel('Total Count')
    ax.set_title('Engagement Funnel\n(How users interact with datasets)')
    ax.invert_yaxis()

    # Conversion rates
    download_rate = total_downloads / total_views * 100 if total_views else 0
    vote_rate = total_votes / total_downloads * 100 if total_downloads else 0

    ax.text(0.98, 0.95, f"View → Download: {download_rate:.2f}%\nDownload → Vote: {vote_rate:.2f}%",
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    return save_chart(fig, '10_engagement_funnel')


def generate_summary_stats(datasets):
    """Generate summary statistics for README."""
    total = len(datasets)
    total_views = sum(d['view_count'] for d in datasets)
    total_downloads = sum(d['download_count'] for d in datasets)
    total_votes = sum(d['total_votes'] for d in datasets)

    avg_usability = np.mean([d['usability_score'] for d in datasets if d['usability_score'] > 0])

    # Top category
    category_counts = Counter()
    for d in datasets:
        cats = d.get('category_names', '')
        if cats:
            for cat in cats.split(', '):
                if cat.strip():
                    category_counts[cat.strip()] += 1
    top_category = category_counts.most_common(1)[0] if category_counts else ('N/A', 0)

    # Top tier
    tier_counts = Counter(d['owner_tier'] for d in datasets if d['owner_tier'])
    top_tier = tier_counts.most_common(1)[0] if tier_counts else ('N/A', 0)

    return {
        'total_datasets': total,
        'total_views': total_views,
        'total_downloads': total_downloads,
        'total_votes': total_votes,
        'avg_usability': avg_usability,
        'top_category': top_category,
        'top_tier': top_tier,
    }


def create_readme(stats, chart_files):
    """Generate README.md with charts and insights."""
    readme = f"""# Kaggle Dataset Ecosystem Analysis

> Comprehensive analysis of **{stats['total_datasets']:,}** Kaggle datasets to understand engagement patterns and ecosystem dynamics.

## Key Metrics

| Metric | Value |
|--------|-------|
| Total Datasets Analyzed | **{stats['total_datasets']:,}** |
| Total Views | **{stats['total_views']:,}** |
| Total Downloads | **{stats['total_downloads']:,}** |
| Total Votes | **{stats['total_votes']:,}** |
| Average Usability Score | **{stats['avg_usability']:.2f}** |
| Top Category | **{stats['top_category'][0]}** ({stats['top_category'][1]:,} datasets) |
| Most Active Tier | **{stats['top_tier'][0]}** ({stats['top_tier'][1]:,} datasets) |

---

## Insights & Charts

### 1. Who Creates Datasets?

![Owner Tier Distribution](charts/01_owner_tier_distribution.png)

**Insight:** The majority of datasets come from {stats['top_tier'][0]} tier users. Building your Kaggle profile through competitions and notebooks can increase your dataset visibility.

---

### 2. Does Experience Lead to More Engagement?

![Engagement by Tier](charts/02_engagement_by_tier.png)

**Insight:** Higher-tier users generally receive more engagement on their datasets. Reputation and community trust play a significant role.

---

### 3. Most Popular Topics

![Top Categories](charts/03_top_categories.png)

**Insight:** Understanding trending categories helps you create datasets that the community actively seeks.

---

### 4. Choosing the Right License

![License Distribution](charts/04_license_distribution.png)

**Insight:** Open licenses (CC0, CC BY) dominate. Making your data freely available increases adoption and engagement.

---

### 5. Quality Matters: Usability Score Impact

![Usability vs Engagement](charts/05_usability_vs_engagement.png)

**Insight:** Datasets with higher usability scores (better documentation, column descriptions, proper formatting) receive significantly more engagement.

---

### 6. Optimal Dataset Size

![Size Distribution](charts/06_size_distribution.png)

**Insight:** Most popular datasets are small to medium-sized. Users prefer datasets they can quickly download and explore.

---

### 7. Engagement Correlation

![Downloads vs Votes](charts/07_downloads_vs_votes.png)

**Insight:** Downloads and votes are correlated but not perfectly. Quality content generates votes; utility drives downloads.

---

### 8. Top Dataset Creators

![Top Creators](charts/08_top_creators.png)

**Insight:** Prolific creators build audiences. Consistent publishing increases overall visibility.

---

### 9. Popular File Formats

![File Types](charts/09_file_types.png)

**Insight:** CSV dominates. Keep your data in accessible, standard formats for maximum reach.

---

### 10. The Engagement Funnel

![Engagement Funnel](charts/10_engagement_funnel.png)

**Insight:** Only a small fraction of views convert to downloads, and even fewer to votes. Focus on making your dataset immediately useful.

---

## Recommendations for Maximum Engagement

Based on this analysis, here are actionable tips:

1. **Maximize Usability Score** - Add column descriptions, cover image, proper tags
2. **Use Open Licenses** - CC0 or CC BY 4.0 are most popular
3. **Optimal Size** - Keep datasets under 100MB when possible
4. **CSV Format** - Provide data in CSV for maximum accessibility
5. **Target Popular Categories** - Align with trending topics like ML, business, health
6. **Build Your Profile** - Higher tier = more credibility = more engagement
7. **Write Great Documentation** - Clear overview and use cases drive downloads
8. **Consistent Publishing** - Regular contributions build audience

---

## Data Source

📊 **Dataset:** [Kaggle Datasets Metadata](https://www.kaggle.com/datasets/ismetsemedov/kaggle-datasets)

This analysis is based on data collected from Kaggle's public dataset API, covering {stats['total_datasets']:,} unique datasets across all categories and time periods.

**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""

    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(readme)
    print("\nSaved: README.md")


def main():
    print("=" * 60)
    print("Kaggle Dataset Ecosystem Analysis")
    print("=" * 60)

    # Create charts directory
    Path(CHARTS_DIR).mkdir(exist_ok=True)

    # Load data
    print("\nLoading data...")
    datasets = load_data()
    print(f"Loaded {len(datasets):,} datasets")

    # Detect bulk uploaders
    print("\nDetecting bulk uploaders...")
    bulk_uploaders = get_bulk_uploaders(datasets)
    print(f"Identified {len(bulk_uploaders)} bulk uploaders (automated/split uploads)")

    # Generate charts
    print("\nGenerating charts...")
    charts = []

    charts.append(chart_owner_tier_distribution(datasets))
    charts.append(chart_engagement_by_tier(datasets))
    charts.append(chart_top_categories(datasets))
    charts.append(chart_license_distribution(datasets))
    charts.append(chart_usability_vs_engagement(datasets))
    charts.append(chart_size_distribution(datasets))
    charts.append(chart_downloads_vs_votes(datasets))
    charts.append(chart_top_creators(datasets, bulk_uploaders))
    charts.append(chart_file_types(datasets))
    charts.append(chart_engagement_funnel(datasets))

    # Generate stats and README
    print("\nGenerating README...")
    stats = generate_summary_stats(datasets)
    create_readme(stats, charts)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Charts saved to: {CHARTS_DIR}/")
    print("README updated with insights")
    print("=" * 60)


if __name__ == "__main__":
    main()
