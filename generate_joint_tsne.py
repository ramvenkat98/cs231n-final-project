import pickle

model_name_1 = "Baseline"
embeddings_filename_1 = 'embeddings_for_unfinetuned_model.pkl'
model_name_2 = "Fine-tuned (line + bar)"
embeddings_filename_2 = 'embeddings_for_pure_bar_chart_finetuned.pkl'

categories_we_want = ['bar chart', 'line plot']


with open(embeddings_filename_1, 'rb') as f:
  classes_to_embeddings_1_means = pickle.load(f)

with open(embeddings_filename_2, 'rb') as f:
  classes_to_embeddings_2_means = pickle.load(f)

classes_to_embeddings_means = {}
for category in categories_we_want:
    classes_to_embeddings_means[f'{category}: {model_name_1.lower()}'] = classes_to_embeddings_1_means[category]
    classes_to_embeddings_means[f'{category}: {model_name_2.lower()}'] = classes_to_embeddings_2_means[category]

for x in classes_to_embeddings_means.keys(): print(x, len(classes_to_embeddings_means[x]))
# print(classes_to_embeddings_means['bar chart'][0].shape)
'''
classes_to_embeddings_means['synthetic line plot (val)'] = classes_to_embeddings_means['Augmented Line Charts (Val)']
classes_to_embeddings_means['synthetic bar chart (val)'] = classes_to_embeddings_means['Augmented Bar Charts (Val)']

del classes_to_embeddings_means['Augmented Bar Charts (Val)']
del classes_to_embeddings_means['Augmented Bar Charts (Train)']
del classes_to_embeddings_means['Augmented Line Charts (Val)']
del classes_to_embeddings_means['Augmented Line Charts (Train)']
'''

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

plt.title(f"Joint t-SNE visualization of {model_name_1} and {model_name_2} embeddings")
plt.xlabel("First t-SNE")
plt.ylabel("Second t-SNE")
colorbar = plt.colorbar(scatter, ticks=[i for i in range(len(label_map))])
colorbar.set_ticklabels([inverse_label_map[i] for i in range(len(label_map))])

plt.show()
