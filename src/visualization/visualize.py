# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('paper')

# score and ranking
sns.scatterplot(x=teams_df.Score, y=teams_df.Ranking)

# correlation coefficient matrix
corr = df.iloc[:,:40].corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(35, 30))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(240, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
corr_heatmap = sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
            annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
corr_heatmap_fig = corr_heatmap.get_figure()    
corr_heatmap_fig.savefig('svm_conf.png', dpi=100)
