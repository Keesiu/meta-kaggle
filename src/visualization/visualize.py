# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.feature_selection import f_regression, SelectKBest

sns.set_context('paper')

# score and ranking
sns.scatterplot(x=y.Score, y=y.Ranking)

# histogram
sns.distplot(y.Score)
sns.distplot(y.Ranking)

# pairplot
sns.pairplot(pd.concat([y, X], axis=1))

# correlation coefficient matrix
corr = X.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(15, 15))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(240, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
corr_heatmap = sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
corr_heatmap_fig = corr_heatmap.get_figure()    
corr_heatmap_fig.savefig('corr_heatmap_after_vif.png', dpi=100)

# plot regression model
sns.regplot(x='radon_avg_cc', y=y.score, data=X, logistic=True)

#%% exploratory analysis

# correlation coefficient matrix
corr = X.corr()

# compute Pearson correlation coefficient and p-value for each pair
# (not reliable for small datasets)
corr_score = pd.DataFrame(
        data = [list(pearsonr(X[x], y.score)) for x in X],
        index = X.columns,
        columns = ['corr_coeff', 'p_value'])
corr_ranking_log = pd.DataFrame(
        data = [list(pearsonr(X[x], y.ranking_log)) for x in X],
        index = X.columns,
        columns = ['corr_coeff', 'p_value'])

# compute f_scores
f_score, pval = f_regression(X, y.score)    
f_score_df = pd.DataFrame(
        data = {'f_score' : f_score,
                'pval' : pval},
        index = X.columns)
f_score, pval = f_regression(X, y.ranking_log)
f_ranking_log_df = pd.DataFrame(
        data = {'f_score' : f_score,
                'pval' : pval},
        index = X.columns)
del f_score, pval