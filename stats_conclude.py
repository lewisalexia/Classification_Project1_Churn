from scipy import stats

# STATS CONCLUSIONS FUNCTIONS 


def chi2_test(df):
    """This function sets alpha to 0.05, creates a table, runs a chi^2 test, 
    and evaluates the pvalue against alpha, printing a conclusion statement.
    """
    import pandas as pd
    α = 0.05
    table = pd.crosstab(df.phone_service, 
                       df['internet_service_type_Fiber optic'])
    chi2, pval, degf, expected = stats.chi2_contingency(table)
    # print('Observed')
    # print(table.values)
    # print('\nExpected')
    # print(expected.astype(int))
    # print('\n----')
    # print(f'chi^2 = {chi2:.4f}')
    # print(f'p-value = {pval} < {α}')
    # print('----')
    if pval < α:
        print ('We reject the null hypothesis. Phone and Fiber are related.')
    else:
        print ("We fail to reject the null hypothesis.")





def conclude_1samp_tt(group1, group_mean):
    """This function sets alpha to 0.05, runs a one sample, two tailed test, 
    and evaluates the pvalue and tstat against alpha, printing a conclusion statement.
    
    This function takes in the main population and the mean of the sample population.
    """
    from scipy import stats
    α = 0.05
    tstat, p = stats.ttest_1samp(group1, group_mean)
    print(f't-stat: {tstat} > 0?')
    print(f'p-value: {p} < {α}?')
    print('\n----')
    if ((p < α) & (tstat > 0)):
        print("we can reject the null hypothesis.")
    else:
        print('We fail to reject the null hypothesis.')




def conclude_1samp_gt(group1, group_mean):
    """This function sets alpha to 0.05, runs a one sample, one tailed test (greater than), 
    and evaluates the pvalue and tstat against alpha, printing a conclusion statement.
    
    This function takes in the main population and the mean of the sample population.
    """
    from scipy import stats
    α = 0.05
    tstat, p = stats.ttest_1samp(group1, group_mean)
    print(f't-stat: {tstat} > 0?')
    print(f'p-value: {p/2} < {α}?')
    print('\n----')
    if ((p / 2) < α) and (tstat > 0):
        print("we can reject the null hypothesis.")
    else:
        print('We fail to reject the null hypothesis.')





def conclude_1samp_lt(group1, group_mean):
    """This function sets alpha to 0.05, runs a one sample, one tailed test (less than), 
    and evaluates the pvalue and tstat against alpha, printing a conclusion statement.
    
    This function takes in the main population and the mean of the sample population.
    """
    from scipy import stats
    α = 0.05
    tstat, p = stats.ttest_1samp(group1, group_mean)
    print(f't-stat: {tstat} < 0?')
    print(f'p-value: {p/2} < {α}?')
    print('\n----')
    if ((p / 2) < α) and (tstat < 0):
        print("we can reject the null hypothesis.")
    else:
        print('We fail to reject the null hypothesis.')






def conclude_2samp_tt(sample1, sample2):
    """This function sets alpha to 0.05, defaults equal_var to True, runs a 
    two sample, two tailed test, and evaluates the pvalue against alpha, printing 
    a conclusion statement.
    
    This function takes in two seperate populations.
    """
    from scipy import stats
    α = 0.05
    tstat, p = stats.ttest_ind(sample1, sample2, equal_var=True)
    print(f't-stat: {tstat}')
    print(f'p-value: {p} < {α}?')
    print('\n----')
    if p < α:
        print("we can reject the null hypothesis.")
    else:
        print('We fail to reject the null hypothesis.')




def conclude_2samp_gt(sample1, sample2):
    """This function sets alpha to 0.05, defaults equal_var to True, runs a two 
    sample, one tailed test (greater than), and evaluates the pvalue and tstat 
    against alpha, printing a conclusion statement.
    
    This function takes in two seperate populations.
    """
    from scipy import stats
    α = 0.05
    tstat, p = stats.ttest_ind(sample1, sample2, equal_var=True)
    print(f't-stat: {tstat} > 0?')
    print(f'p-value: {p/2} < {α}?')
    print('\n----')
    if (((p/2) < α) and (tstat > 0)):
        print("we can reject the null hypothesis.")
    else:
        print('We fail to reject the null hypothesis.')




def conclude_2samp_lt(sample1, sample2):
    """This function sets alpha to 0.05, defaults equal_var to True, runs a two 
    sample, one tailed test (less than), and evaluates the pvalue and tstat 
    against alpha, printing a conclusion statement.
    
    This function takes in two seperate populations.
    """
    from scipy import stats
    α = 0.05
    tstat, p = stats.ttest_ind(sample1, sample2, equal_var=True)
    print(f't-stat: {tstat} < 0?')
    print(f'p-value: {p/2} < {α}?')
    print('\n----')
    if (((p/2) < α) and (tstat < 0)):
        print("we can reject the null hypothesis.")
    else:
        print('We fail to reject the null hypothesis.')





def conclude_anova(theoretical_mean, group1, group2):
    """This function sets alpha to 0.05, runs an ANOVA test, and evaluates the 
    pvalue and tstat against alpha, printing a conclusion statement.
    
    This function takes in the theoretical mean (average of pop), and two independent
    groups.
    """
    from scipy import stats
    α = 0.05
    tstat, pval = stats.f_oneway(theoretical_mean, group1, group2)
    print(f't-stat: {tstat}')
    print(f'p-value: {p} < {α}?')
    print('----')
    if pval < α:
        print("We can reject the null hypothesis.")
    else:
        print('We fail to reject the null hypothesis.')






def conclude_pearsonr(floats1, floats2):
    """This function sets alpha to 0.05, runs a Pearson's R Correlation test on 
    parametric data, evaluates the pvalue against alpha, and prints a conclusion 
    statement.
    
    This function takes in two seperate floats and is used when the relationship is 
    linear, both variables are quantitative, normally distributed, and have no outliers.
    """
    from scipy import stats
    α = 0.05
    r, p = stats.pearsonr(floats1, floats2)
    print(f'r (correlation value): {r}')
    print(f'p-value: {p} < {α}?')
    print('----')
    if p < α:
        print("We can reject the null hypothesis.")
    else:
        print("We fail to reject the null hypothesis.")




def conclude_spearmanr(floats1, floats2):
    """This function sets alpha to 0.05, runs a Spearman's Correlation test on non-
    parametric data, evaluates the pvalue against alpha, and prints a conclusion
    statement.
    
    This function takes in two seperate floats and is used when the relationship is 
    rank-ordered, both variables are quantitative, NOT normally distributed, and
    presents as "might be monotonic, might be linear" through a scatterplot visual.
    """
    from scipy import stats
    α = 0.05
    r, p = stats.spearmanr(floats1, floats2)
    print(f'r (correlation value): {r}')
    print(f'p-value: {p} < {α}?')
    print('----')
    if p < α:
        print("We can reject the null hypothesis.")
    else:
        print("We fail to reject the null hypothesis.")




def conclude_mannwhitneyu(subpop1, subpop2):
    """This function sets alpha to 0.05, runs a Mann-Whitney U test on non-parametric
    data, evaluates the pvalue against alpha, and prints a conclusion statement.
    
    This function takes in two sub-populations and is usually used to compare
    sample means: use when the data is ordinal (non-numeric) and t-test assumptions
    are not met.
    """
    from scipy import stats
    α = 0.05
    t, p = stats.mannwhitneyu(subpop1, subpop2)
    print(f"t-stat: {t}")
    print(f'p-value: {p} < {α}?')
    if p < α:
        print("We reject the null hypothesis.")
    else:
        print("We fail to reject the null hypothesis.")