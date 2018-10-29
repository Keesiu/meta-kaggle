# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

sns.set_context('paper')

# score and ranking
sns.scatterplot(x=y.ranking_log, y=y.score)

# histogram
sns.distplot(y.score, bins=50)
sns.distplot(y.ranking_log)
sns.distplot(X.loc_max_log)
sns.distplot(X.pylint_warning_ratio)
sns.distplot(X.radon_h_effort_ratio)
sns.distplot(X.radon_mi_mean)
sns.distplot(X.pylint_)
sns.distplot(X.loc_max_log)
# pairplot
sns.pairplot(pd.concat([y, X], axis=1))

# boxplot
sns.boxplot(data=X)

# residual plot
sns.residplot(x=mod_sm.fittedvalues, y=y, data=X, lowess=True)

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
#corr_heatmap_fig = corr_heatmap.get_figure()    
#corr_heatmap_fig.savefig('corr_heatmap_after_vif.png', dpi=100)

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

#%% plot ElasticNetCV results

    mse_path = pd.DataFrame(data=mod.mse_path_, index=ALPHAS)
    logger.info("Lasso MSE = {}.".format(mse_path))
    # Display results
    m_log_alphas = -np.log(mod.alphas_)/np.log(BASE)
    plt.figure(figsize=(10,8))
    ymin, ymax = 0, 1000
    plt.plot(m_log_alphas, mod.mse_path_, ':')
    plt.plot(m_log_alphas, mod.mse_path_.mean(axis=-1), 'k',
             label='Average across the folds', linewidth=2)
    plt.axvline(-np.log10(mod.alpha_), linestyle='--', color='k',
                label='alpha: CV estimate')
    plt.legend()
    plt.xlabel('-log(alpha)')
    plt.ylabel('Mean square error')
    plt.title('Mean square error on each fold')
    plt.axis('tight')
    plt.ylim(ymin, ymax)
    plt.show()