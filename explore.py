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

def explore(df):
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
    target = 'churn' # assigning target variable
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


# assigning features to train, validate, and test for modeling
def define_telco(train, validate, test, target):
    """This function takes in the train, validate, and test dataframes and assigns 
    the chosen features to X_train, X_validate, X_test, and y_train, y_validate, 
    and y_test.
    """
    # assign target feature
    target = 'churn'

    # X_train, validate, and test to be used for modeling
    variable_list = []
    X_train = train[[
    'phone_service',
    'multiple_lines',
    'monthly_charges',
    'total_charges',
    'contract_type_One year',
    'contract_type_Two year',
    'internet_service_type_Fiber optic',
    'internet_service_type_None']]
    variable_list.append(X_train)
    X_validate = validate[[
    'phone_service',
    'multiple_lines',
    'monthly_charges',
    'total_charges',
    'contract_type_One year',
    'contract_type_Two year',
    'internet_service_type_Fiber optic',
    'internet_service_type_None']]
    variable_list.append(X_validate)
    X_test = test[[
    'phone_service',
    'multiple_lines',
    'monthly_charges',
    'total_charges',
    'contract_type_One year',
    'contract_type_Two year',
    'internet_service_type_Fiber optic',
    'internet_service_type_None']]
    variable_list.append(X_test)
    y_train = train[target]
    variable_list.append(y_train)
    y_validate = validate[target]
    variable_list.append(y_validate)
    y_test = test[target]
    variable_list.append(y_test)

    return variable_list

    
    

def total_monthly(df):
    """This function displays monthly charges causing churn as they increase.
    """
    sns.scatterplot(data=df, x=df.monthly_charges, \
    y=df.total_charges, hue=df.churn)
    plt.axvline(x=df.monthly_charges.mean(), label='Average Monthly Charge', color='red', linewidth=2)
    plt.title('High Monthly Charges Creates Churn')
    plt.legend()
    plt.show()



def box_plot_monthly_multiple(df):
    """This function displays that mean monthly charges are higher for churned
    customers with and without multiple lines but ESPECIALLY for multiple line
    customers.
    """
    sns.boxplot(x=df.multiple_lines, \
                y=df.monthly_charges, hue=df.churn)
    plt.title('Houston, We Have a Problem...')
    plt.axhline(y=df.monthly_charges.mean(), label='Average Monthly Charge', color='red', linewidth=2)
    plt.xticks([0, 1], ['No', 'Yes'])
    plt.legend([0, 1], ['No', 'Yes'])
    plt.show()


def phone_fiber(df):
    """This function displays that all fiber customers have phone service.
    """
    sns.barplot(data=df, x='phone_service', y='internet_service_type_Fiber optic')
    plt.title('All Fiber Customers Have Phone Service')
    plt.xticks([0, 1], ['No', 'Yes'])
    plt.ylabel('Fiber Internet')
    plt.xlabel('Phone Service')
    plt.show()

def monthly_phone_churn(df):
    """This function displays the churn rate of customers with one or more lines of phone
    service.
    """
    sns.barplot(data=df, x='churn', \
    y='contract_type', hue='multiple_lines')
    plt.axvline(x=df['churn'].mean(), label='Average Churn Rate', color='red', linewidth=2)
    plt.title('Phone Contracts Well Above Average Churn Rate')
    plt.legend()
    plt.show()


def fiber_average_cost(df):
    """This function displays churn customers being charged more than average monthly price.
    """
    sns.scatterplot(data=df, x='monthly_charges', y='total_charges', hue='phone_service')
    plt.axvline(x=df['monthly_charges'].mean(), label='Average Churn Rate', color='red', linewidth=2)
    plt.title('There Is Wiggle Room To Reduce Price')
    plt.show()

def monthly_contract(df):
    """This function displays the slope of charge for phone customers. The purpose is to
    show there is room to level the price.
    """
    sns.scatterplot(data=df, x='monthly_charges', y='total_charges', hue='contract_type')
    plt.title('Monthly Contracts Make Up Majority of Customer Base')
    plt.xlabel('Monthly Charges')
    plt.ylabel('Total Charges')
    plt.show()