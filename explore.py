import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import stats_conclude as sc
from scipy import stats

from sklearn.metrics import classification_report, confusion_matrix

import warnings
warnings.filterwarnings("ignore")

# EXPLORES CATEGORICAL AND NUMERICAL VARIABLES. PROVIDES GRAPHS, STATS TEST FOR
# CATEGORICALS AND VISUALS FOR NUMERICALS.

def explore(df, target):
    '''This function takes in a df and a defined target variable to explore.
    
    This function is meant to assign the df columns to categorical and numerical 
    columns. The default for numerical is more than 3 uniques assigns it to a list of
    numericals (col_num). Less than 3 indicates "buckets" which indicates a categorical 
    variable (col_cat).

    The function will then print for each col in col_cat:
        * Value Counts
        * Proportional size of data
        * Hypotheses (null + alternate)
        * Analysis and summary using CHI^2 test function (chi2_test from acquire.py)
        * A conclusion statement
        * A graph representing findings

    The function will then print for each col in col_num:
        * A graph with the mean compared to the target variable outcomes.
            * Take the output from col_num and move forward into the 
              appropriate statistics tests.
    '''
    col_cat = [] #this is for my categorical varibles
    col_num = [] #this is for my numerical varibles
    for col in df.columns:
        if len(df[col].unique()) < 3: # making anything with less than 3 unique variables a categorical
            col_cat.append(col)
        else:
            col_num.append(col)
    for col in col_cat:
        print(f"CATEGORICAL VARIABLE\n**{col.upper()}**")
        print(df[col].value_counts())
        print(df[col].value_counts(normalize=True)*100)
        print()
        print(f'HYPOTHESIZE')
        print(f"H_0: {col.lower().replace('_',' ')} does not affect {target}")
        print(f"H_a: {col.lower().replace('_',' ')} affects {target}")
        print()
        print('ANALYZE and SUMMARIZE')
        observed = pd.crosstab(df[col], df[target])
        α = 0.05
        chi2, pval, degf, expected = stats.chi2_contingency(observed)
        print('Observed')
        print(observed.values)
        print('\nExpected')
        print(expected.astype(int))
        print('\n----')
        print(f'chi^2 = {chi2:.4f}')
        print(f'p-value = {pval} < {α}')
        print('----')
        if pval < α:
            print ('We reject the null hypothesis.')
        else:
            print ("We fail to reject the null hypothesis.")
        print()
        print(f'VISUALIZE')
        sns.barplot(x=df[col], y=df[target])
        plt.title(f"{col.lower().replace('_',' ')} vs {target}")
        plt.show()
        print(f'\n')

    for col in col_num:
        print(f"NUMERICAL VARIABLE")
        print(f"If deemed significant, proceed with the appropriate statistics tests.")
        sns.barplot(data=df, x=df[target], y=df[col])
        plt.title(f"Is {target} independent of {col.lower().replace('_',' ')}?")
        pop_mn = df[col].mean()
        plt.axhline(pop_mn, label=(f"{col.lower().replace('_',' ')} mean"))
        plt.legend()
        plt.show()
        print()