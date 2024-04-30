import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.lines as mlines
# Set Arial as the default font
import argparse

parser = argparse.ArgumentParser(description='Main entrance of AGE')
parser.add_argument('--data', type=str, default='AGE',
                    help='Baron_mouse.csv')
args = parser.parse_args()
labels_true = pd.read_csv('Baron_human_cell.csv')
labels_true = np.array(labels_true)
labels_true = labels_true[:, 1].tolist()

#data = pd.read_csv(ndata + '/Baron_human_embedding.csv')
data = pd.read_csv('Baron_human_embedding.csv')
data = np.array(data)
data = data[:, 1:]
color = labels_true

    # color = color[:, 4]
    # color = color[:, 1]-1
L = max(color)
    # Define the label names
    # label_names = ['Tumor', 'B cell', 'Dendritic', 'Endothelial', 'Fibroblast',
    #                'Macrophage', 'Mast', 'myocyte', 'T cell', 'undefined']
    # label_names = ['d0', 'd2', 'd4', 'd7']
    # label_names = ['Bcell', 'Myeloid', 'Tcell']
label_Baron_mouse = ["beta","ductal","delta","schwann","quiescent_stellate","endothelial","gamma","alpha",
    "macrophage","immune_other","activated_stellate","B_cell","T_cell"]
label_Baron_human = ["beta","delta","t_cell","activated_stellate","ductal",
                        "alpha","endothelial","epsilon","quiescent_stellate",
                        "mast","macrophage","mast","schwann","gamma"
    ]
label_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10', '11', '12','13']
label_map = dict(zip(label_names, range(len(label_names))))

    # Create a list of labels that correspond to the color values
labels = [label_names[i] for i in color]

embedding = TSNE(random_state=42).fit_transform(data)


    # Create a scatter plot of the embedding with color-coded labels
fig, ax = plt.subplots(figsize=(100, 80))
fig, ax = plt.subplots(figsize=(120, 80))
scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap='Spectral', s=600)

    # Set the aspect ratio of the plot to equal and set axis limits
ax.set_aspect('equal', 'datalim')
ax.set_xlim((-80, 80))
ax.set_ylim((-80, 60))
    # Set font properties
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.weight"] = "bold"

    # Set tick font size and weight
ax.tick_params(axis='both', which='major', labelsize=100, width=5, length=20)
ax.tick_params(axis='both', which='minor', labelsize=50, width=5, length=10)
ax.set_ylabel('Baron_mouse', fontsize=200, fontweight='bold')
ax.set_title(args.data, fontsize=400, fontweight='bold')



handles = []
labels = []
for i, label_name in enumerate(label_Baron_mouse):
        handles.append(mlines.Line2D([], [], color=scatter.cmap(scatter.norm(np.array([i]))), marker='o', linestyle='None', markersize=50))
        labels.append(label_name)


    # Set border width
for spine in ax.spines.values():
        spine.set_linewidth(10)

legend1 = ax.legend(handles, labels, prop={'size': 150, 'weight': 'normal'}, markerfirst=True, loc='upper left', borderaxespad=0)
legend1.set_frame_on(False)
ax.add_artist(legend1)
    # Save the plot as a PNG file
#fig.savefig(ndata+"_Baron_human.png")
fig.savefig("_Baron_human.png")



    # plt.show()
print("done")