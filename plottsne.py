import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.lines as mlines
# Set Arial as the default font

labels_true = pd.read_csv('count/HNSCC/HNSCC_cell.csv')
labels_true = np.array(labels_true)
labels_true = labels_true[:, 1].tolist()

labels_pred = pd.read_csv('count/HNSCC/HNSCC_results.csv')
labels_pred = np.array(labels_pred)
labels_pred = labels_pred[:, 1].tolist()

data = pd.read_csv('count/HNSCC/HNSCC_embedding.csv')
data = np.array(data)
data = data[:, 1:]
color = labels_true

# color = color[:, 4]
# color = color[:, 1]-1
L = max(color)
# Define the label names
label_names = ['Tumor', 'B cell', 'Dendritic', 'Endothelial', 'Fibroblast',
               'Macrophage', 'Mast', 'myocyte', 'T cell', 'undefined']
label_map = dict(zip(label_names, range(len(label_names))))

# Create a list of labels that correspond to the color values
labels = [label_names[i] for i in color]

embedding = TSNE(random_state=6).fit_transform(data[:, 1:])


# Create a scatter plot of the embedding with color-coded labels
fig, ax = plt.subplots(figsize=(60, 45))
scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap='Spectral', s=80)

# Set the aspect ratio of the plot to equal and set axis limits
ax.set_aspect('equal', 'datalim')
# ax.set_xlim((-10, 50))
# ax.set_ylim((-10, 50))
# Set font properties
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.weight"] = "bold"

# Set tick font size and weight
ax.tick_params(axis='both', which='major', labelsize=20, width=2, length=10)
ax.tick_params(axis='both', which='minor', labelsize=20, width=2, length=5)
ax.set_ylabel('HNSCC', fontsize=100, fontweight='bold')
ax.set_title('t-SNE Embedding of scDAGAE', fontsize=120, fontweight='bold')



handles = []
labels = []
for i, label_name in enumerate(label_names):
    handles.append(mlines.Line2D([], [], color=scatter.cmap(scatter.norm(np.array([i]))), marker='o', linestyle='None', markersize=50))
    labels.append(label_name)


# Set border width
for spine in ax.spines.values():
    spine.set_linewidth(5)

legend1 = ax.legend(handles, labels, loc="upper left", prop={'size': 50, 'weight': 'bold'}, markerfirst=True)

ax.add_artist(legend1)
# Save the plot as a PNG file
fig.savefig("HNSCC_plot.png")
print("done")