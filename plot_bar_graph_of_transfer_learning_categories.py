'''
import matplotlib.pyplot as plt

categories = ['geometry', 'scatter plot', 'table', 'puzzle']
scores_baseline = [96 / 203, 11 / 16, 6 / 13, 5 / 23]
scores_train_line_only = [112 / 203, 8 / 16, 7 / 13, 3 / 23]
scores_train_bar_only = [112 / 203, 7 / 16, 6 / 13, 1 / 23]
scores_train_line_and_bar = [110 / 203, 8 / 16, 6 / 13, 2 / 23]
'''

import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ['Geometry', 'Scatter Plot', 'Table', 'Puzzle']
scores_baseline = [96 / 203, 11 / 16, 6 / 13, 5 / 23]
scores_train_line_only = [112 / 203, 8 / 16, 7 / 13, 3 / 23]
scores_train_bar_only = [112 / 203, 7 / 16, 6 / 13, 1 / 23]
scores_train_line_and_bar = [110 / 203, 8 / 16, 6 / 13, 2 / 23]

# Convert scores to percentages
scores_baseline = [x * 100 for x in scores_baseline]
scores_train_line_only = [x * 100 for x in scores_train_line_only]
scores_train_bar_only = [x * 100 for x in scores_train_bar_only]
scores_train_line_and_bar = [x * 100 for x in scores_train_line_and_bar]

# X locations for the groups
ind = np.arange(len(categories))
width = 0.2  # Width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind - 1.5 * width, scores_baseline, width, label='Baseline')
rects2 = ax.bar(ind - 0.5 * width, scores_train_line_only, width, label='Train Line Only')
rects3 = ax.bar(ind + 0.5 * width, scores_train_bar_only, width, label='Train Bar Only')
rects4 = ax.bar(ind + 1.5 * width, scores_train_line_and_bar, width, label='Train Bar and Line')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores (%)')
ax.set_title('Scores by Model and Category')
ax.set_xticks(ind)
ax.set_xticklabels(categories)
ax.legend()

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(int(height)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# Automatically label each bar with height
autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

fig.tight_layout()

plt.show()

