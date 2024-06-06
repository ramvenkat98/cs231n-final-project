import pickle

# model_name = "Baseline"
# embeddings_filename = 'embeddings_for_unfinetuned_model.pkl'
model_name = "Fine-tuned (line + bar)"
embeddings_filename = 'embeddings_for_bar_chart_and_line_chart_finetuned.pkl'
with open(embeddings_filename, 'rb') as f:
  classes_to_embeddings_means = pickle.load(f)

for x in classes_to_embeddings_means.keys(): print(x, len(classes_to_embeddings_means[x]))
print(classes_to_embeddings_means['bar chart'][0].shape)

classes_to_embeddings_means['synthetic line plot (val)'] = classes_to_embeddings_means['Augmented Line Charts (Val)']
classes_to_embeddings_means['synthetic bar chart (val)'] = classes_to_embeddings_means['Augmented Bar Charts (Val)']

del classes_to_embeddings_means['Augmented Bar Charts (Val)']
del classes_to_embeddings_means['Augmented Bar Charts (Train)']
del classes_to_embeddings_means['Augmented Line Charts (Val)']
del classes_to_embeddings_means['Augmented Line Charts (Train)']

embeddings = []
class_labels = []
for x in classes_to_embeddings_means:
  class_labels += [x] * len(classes_to_embeddings_means[x])
  embeddings += classes_to_embeddings_means[x]

print(len(embeddings), len(class_labels))

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import numpy as np

embeddings = np.array([x for x in embeddings])
scaler = StandardScaler()
embeddings_norm = scaler.fit_transform(embeddings)

scaler.mean_.shape

tsne = TSNE(n_components=2, random_state=42, perplexity = 50)
X_tsne = tsne.fit_transform(embeddings_norm)
print(tsne.kl_divergence_)
print(X_tsne.shape)

label_map = {}
inverse_label_map = {}
for (i, x) in enumerate(classes_to_embeddings_means.keys()):
  label_map[x] = i
  inverse_label_map[i] = x

class_indices = [label_map[x] for x in class_labels]

import matplotlib.pyplot as plt

# Assuming X_tsne is your t-SNE data and y is your labels
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=class_indices, cmap = 'inferno')

plt.title(f"t-SNE visualization of {model_name}")
plt.xlabel("First t-SNE")
plt.ylabel("Second t-SNE")
colorbar = plt.colorbar(scatter, ticks=[i for i in range(len(label_map))])
colorbar.set_ticklabels([inverse_label_map[i] for i in range(len(label_map))])

plt.show()
