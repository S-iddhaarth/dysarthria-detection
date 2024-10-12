import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_distribution_per_class(
    distribution: dict, nrows: int, ncols: int, figsize: tuple[int, int],
    x_label: str, y_label: str, label_decoding: dict = None, grid: bool = True,
    plot_type: str = 'barplot') -> None:
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()  # Flatten the axes array for easy indexing
    
    labels = list(distribution.keys())
    count = 0
    
    for i in range(nrows):
        for j in range(ncols):
            if count >= len(labels):
                break  # Exit loop if no more labels
            
            ax = axes[count]
            label = labels[count]
            title = label_decoding[label] if label_decoding else label
            try:
                x = list(distribution[label].keys())
                y = list(distribution[label].values())
            except:
                x = distribution[label]
            
            if plot_type == 'barplot':
                sns.barplot(ax=ax, x=x, y=y)
            elif plot_type == 'log_histogram':
                ax.hist(x, weights=y, bins=50, log=True)
            elif plot_type == 'cdf':
                data_sorted = np.sort(x)
                cdf = np.cumsum(y) / np.sum(y)
                ax.plot(data_sorted, cdf, marker='.', linestyle='none')
            elif plot_type == 'boxplot':
                sns.boxplot(ax=ax, x=x)
            elif plot_type == 'violinplot':
                sns.violinplot(ax=ax, x=x)
            elif plot_type == 'kde':
                sns.kdeplot(ax=ax, x=x, weights=y)
            
            ax.set_title(title, fontweight='bold', fontsize=16)
            ax.grid(grid)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            
            count += 1
    
    plt.tight_layout()
    plt.show()