
# EVALUATON AND VISUALIZATION FUNCTIONS



# DECISION TREE



def classifier_tree_eval(X_train, y_train, X_validate, y_validate):
    ''' This function is to calculate the best classifier decision tree model by running 
    a for loop to explore the max depth per default range (1,20).

    The loop then makes a list of lists of all max depth calculations, compares the
    accuracy between train and validate sets, turns to df, and adds a new column named
    difference. The function then calculates the baseline accuracy and plots the
    baseline, and the train and validate sets to identify where overfitting occurs.
    '''
    scores_all=[]
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeClassifier
    import warnings
    warnings.filterwarnings("ignore")
    for x in range(1,11):
        tree = DecisionTreeClassifier(max_depth=x, random_state=123)
        tree.fit(X_train, y_train)
        train_acc = tree.score(X_train, y_train)
        print(f"For depth of {x:2}, the accuracy is {round(train_acc,2)}")
        
        # evaludate on validate set
        validate_acc = tree.score(X_validate, y_validate)

        # append to df scores_all
        scores_all.append([x, train_acc, validate_acc])

        # turn to df
        scores_df = pd.DataFrame(scores_all, columns=['max_depth', 'train_acc', 'validate_acc'])

        # make new column
        scores_df['difference'] = scores_df.train_acc - scores_df.validate_acc

        # sort on difference
        scores_df.sort_values('difference')

        # establish baseline accuracy
    baseline_accuracy = (y_train == 0).mean()
    print()
    print(f'The baseline accuracy is {round(baseline_accuracy,2)}')
          
        # can plot to visulaize
    plt.figure(figsize=(12,8))
    plt.plot(scores_df.max_depth, scores_df.train_acc, label='train', marker='o')
    plt.plot(scores_df.max_depth, scores_df.validate_acc, label='validate', marker='o')
    plt.axhline(baseline_accuracy, linewidth=2, color='black', label='baseline')
    plt.xlabel('Max Depth for Decision Tree')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(1,11, step=1))
    plt.title('Where do I begin to overfit?')
    plt.legend()
    plt.show()

# select a model before the split of the two graphs. A large split indicates overfitting
# when selecing the depth to run with select the point where the difference between
# the train and validate set is the smallest before they seperate.



# RANDOM FOREST


def random_forest_eval(X_train, y_train, X_validate, y_validate):
    ''' This function is to calculate the best random forest decision tree model by running 
    a for loop to explore the max depth per default range (1,20).

    The loop then makes a list of lists of all max depth calculations, compares the
    accuracy between train and validate sets, turns to df, and adds a new column named
    difference. The function then calculates the baseline accuracy and plots the
    baseline, and the train and validate sets to identify where overfitting occurs.
    '''
    scores_all=[]

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    import warnings
    warnings.filterwarnings("ignore")

    for x in range(1,11):
        rf = RandomForestClassifier(random_state = 123,max_depth = x)
        rf.fit(X_train, y_train)
        train_acc = rf.score(X_train, y_train)
        print(f"For depth of {x:2}, the accuracy is {round(train_acc,2)}")
        
        # establish feature importance variable
        important_features = rf.feature_importances_
        
        # evaluate on validate set
        validate_acc = rf.score(X_validate, y_validate)

        # append to df scores_all
        scores_all.append([x, train_acc, validate_acc])

        # turn to df
        scores_df = pd.DataFrame(scores_all, columns=['max_depth', 'train_acc', 'validate_acc'])

        # make new column
        scores_df['difference'] = scores_df.train_acc - scores_df.validate_acc

        # sort on difference
        scores_df.sort_values('difference')

        # establish baseline accuracy
    baseline_accuracy = (y_train == 0).mean()
    print()
    print(f'The baseline accuracy is {round(baseline_accuracy,2)}')
          
        # plot to visulaize train and validate accuracies for best fit
    plt.figure(figsize=(12,8))
    plt.plot(scores_df.max_depth, scores_df.train_acc, label='train', marker='o')
    plt.plot(scores_df.max_depth, scores_df.validate_acc, label='validate', marker='o')
    plt.axhline(baseline_accuracy, linewidth=2, color='black', label='baseline')
    plt.xlabel('Max Depth for Random Forest')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(1,11, step=1))
    plt.title('Where do I begin to overfit?')
    plt.legend()
    plt.show()
    
        # plot feature importance
    plt.figure(figsize=(12,12))
    plt.bar(X_train.columns, important_features)
    plt.title(f"Feature Importance")
    plt.xlabel(f"Features")
    plt.ylabel(f"Importance")
    plt.xticks(rotation = 60)
    plt.show()   


def random_forest_test(test):
    """This function creates the test dataframe and tests the selected model"""
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    import warnings
    warnings.filterwarnings("ignore")

    X_test = test[[
    'phone_service',
    'multiple_lines',
    'monthly_charges',
    'total_charges',
    'contract_type_One year',
    'contract_type_Two year',
    'internet_service_type_Fiber optic',
    'internet_service_type_None']]
    y_test = test['churn']


    print(f"This is the Random Forest Model with a Max Depth of 6\nand ran on the test set.\n")
    rf = RandomForestClassifier(random_state = 123,max_depth = 6)
    rf.fit(X_test, y_test)
    test_acc = rf.score(X_test, y_test)
    print(f"For a depth of 6, the accuracy is {round(test_acc,2)}")
    
    # establish baseline accuracy
    baseline_accuracy = (y_test == 0).mean()
    print()
    print(f'The baseline accuracy is {round(baseline_accuracy,2)}\n')
    
    # classification report
    print(classification_report(y_test, rf.predict(X_test)))
    
    print(f"The model beats on baseline.")

# KNN
def knn_evaluate(X_tr, y_tr, X_va, y_va, nn):
    """This function evaluates the train and validate set on KNN model. This function uses a 
    for loop to a specified end range, 'nn' (exclusive). 

    The function then selects and explicitly identifies the best fit neighbor size by displaying
    graphically as well as explicitly.
    """
    from sklearn.neighbors import KNeighborsClassifier
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    print(f"KNN")
    print(f"The number of features sent in : {len(X_tr.columns)} and are {X_tr.columns.tolist()}.")

    # run for loop and plot
    metrics = []
    for k in range(1, nn):
        
        # make the model
        knn = KNeighborsClassifier(n_neighbors=k, weights='uniform')
        
        # fit the model
        knn.fit(X_tr, y_tr)
        
        # calculate accuracy
        train_score = knn.score(X_tr, y_tr)
        validate_score = knn.score(X_va, y_va)
        
        # append to df metrics
        metrics.append([k, train_score, validate_score])

        # turn to df
        metrics_df = pd.DataFrame(metrics, columns=['k', 'train score', 'validate score'])
      
        # make new column
        metrics_df['difference'] = metrics_df['train score'] - metrics_df['validate score']
    min_diff_idx = np.abs(metrics_df['difference']).argmin()
    n = metrics_df.loc[min_diff_idx, 'k']

    # make plottable df without the difference column
    metrics_plot = metrics_df.drop(columns='difference')
    print(f"{n} is the number of neighbors that produces the best fit model.")
    print(f"The accuracy score for the train model is {round(train_score,2)}.")
    print(f"The accuracy score for the validate model is {round(validate_score,2)}.")
    
    
    # plot the data
    metrics_plot.set_index('k').plot(figsize = (14,12))
    plt.axvline(x=n, color='black', linestyle='--', linewidth=1, label='best fit neighbor size')
    plt.axhline(y=train_score, color='blue', linestyle='--', linewidth=1, label='train accuracy')
    plt.axhline(y=validate_score, color='orange', linestyle='--', linewidth=1, label='validate accuracy')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0,nn,1))
    plt.legend()
    plt.grid()
    plt.show()



# LOGISTIC REGRESSION


def logit_evaluate(x_df, y_s):
    import pandas as pd
    import numpy as np

    import warnings
    warnings.filterwarnings("ignore")

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    """
    This function takes in a DataFrame (train, validate, test) and
    applies a plain Logistic Regression model with no hyperparameters set 
    outside of default.
    """
    #create it
    logit = LogisticRegression(random_state=123)

    #fit it
    logit.fit(x_df, y_s)

    #use it
    score = logit.score(x_df, y_s)
    print(f"The model's accuracy is {round(score,2)}")
    
    #establish series from array of coefficients to print
    coef = logit.coef_
    
    #baseline
    baseline_accuracy = (y_s == 0).mean()
    print(f"The baseline accuracy is {round(baseline_accuracy,2)}.")

    #classification report
    print(classification_report(y_s, logit.predict(x_df)))

    #coef & corresponding columns
    print(f"The coefficents for features are: {coef.round(2)}.\nThe corresponding columns are {x_df.columns.tolist()}.")
   

# RUN ALL 4 MODELS


def all_4_classifiers(train, validate, nn):
    """This function takes in the train, validate, and test dataframes and assigns 
    the chosen features to X_train, X_validate, X_test, and y_train, y_validate, 
    and y_test.

    This function takes in the train and validate datasets, a KNN number to go 
    to (exclusive) and returns models/visuals/explicit statments for decision tree, 
    random forest, knn, and logistic regression.

    Decision Tree:
        * runs for loop to discover best fit "max depth". Default to 10
        * random_state = 123
        * returns visual representing models ran and where overfitting occurs
        * explicitly identifies the baseline and best fit "max depth"
    
    Random Forest:
        * runs for loop to discover best fit "max depth". Default to 10
        * random_state = 123
        * returns visual representing models ran and where overfitting occurs
        * explicitly identifies the baseline and best fit "max depth"
        * visually presents feature importance

    KNN:
        * runs for loop to discover best fit "number of neighbors". nn argument
            is to set the end range for neighbors for loop (exclusive).
        * explicitly identifes the number of features sent in with column names
        * explicitly identifies the best fit number of neighbors
        * explicitly states accuracy scores for train, validate, and baseline
        * visually represents findings and identifies best fit neighbor size

    Logistic Regression:
        * random_seed = 123
        * runs logit on train and vaidate set
        * prints model, baseline accuracy, and a classification report
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import explore as ex
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    import warnings
    warnings.filterwarnings("ignore")

    # assign target feature
    target = 'churn'

    # X_train, X_validate, X_test, y_train, y_validate, and y_test to be used for modeling
    X_train = train[[
    'phone_service',
    'multiple_lines',
    'monthly_charges',
    'total_charges',
    'contract_type_One year',
    'contract_type_Two year',
    'internet_service_type_Fiber optic',
    'internet_service_type_None']]
    X_validate = validate[[
    'phone_service',
    'multiple_lines',
    'monthly_charges',
    'total_charges',
    'contract_type_One year',
    'contract_type_Two year',
    'internet_service_type_Fiber optic',
    'internet_service_type_None']]
    y_train = train[target]
    y_validate = validate[target]

    # DECISION TREE
    print(f"DECISION TREE")
    scores_all=[]
    
    for i in range(1,11):
        tree = DecisionTreeClassifier(max_depth=i, random_state=123)
        tree.fit(X_train, y_train)
        train_acc = tree.score(X_train, y_train)
        print(f"For depth of {i}, the accuracy is {round(train_acc,2)}")
        
        # evaludate on validate set
        validate_acc = tree.score(X_validate, y_validate)

        # append to df scores_all
        scores_all.append([i, train_acc, validate_acc])

        # turn to df
        scores_df = pd.DataFrame(scores_all, columns=['max_depth', 'train_acc', 'validate_acc'])

        # make new column
        scores_df['difference'] = scores_df.train_acc - scores_df.validate_acc
        
        # establish baseline accuracy
    baseline_accuracy = (y_train == 0).mean()
    print()
    print(f'The baseline accuracy is {round(baseline_accuracy,2)}')
          
        # can plot to visulaize
    plt.figure(figsize=(12,8))
    plt.plot(scores_df.max_depth, scores_df.train_acc, label='train', marker='o')
    plt.plot(scores_df.max_depth, scores_df.validate_acc, label='validate', marker='o')
    plt.axhline(baseline_accuracy, linewidth=2, color='black', label='baseline')
    plt.fill_between(scores_df.max_depth, scores_df.train_acc, scores_df.validate_acc, alpha=.4)
    plt.xlabel('Max Depth for Decision Tree')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(1,11, step=1))
    plt.title('Where do I begin to overfit?')
    plt.legend()
    plt.show()
    
    # RANDOM FOREST
    print(f"RANDOM FOREST")
    scores_rf=[]

    for i in range(1,11):
        rf = RandomForestClassifier(random_state = 123,max_depth = i)
        rf.fit(X_train, y_train)
        train_acc_rf = rf.score(X_train, y_train)
        print(f"For depth of {i:2}, the accuracy is {round(train_acc_rf,2)}")
        
        # establish feature importance variable
        important_features = rf.feature_importances_
        
        # evaluate on validate set
        validate_acc_rf = rf.score(X_validate, y_validate)

        # append to rf scores_all
        scores_rf.append([i, train_acc_rf, validate_acc_rf])

        # turn to df
        scores_df2 = pd.DataFrame(scores_rf, columns=['max_depth', 'train_acc_rf', 'validate_acc_rf'])

        # make new column
        scores_df2['difference'] = scores_df2.train_acc_rf - scores_df2.validate_acc_rf

        # sort on difference
        scores_df2.sort_values('difference')

        # print baseline
    print(f'The baseline accuracy is {round(baseline_accuracy,2)}')
          
        # plot to visulaize train and validate accuracies for best fit
    plt.figure(figsize=(12,8))
    plt.plot(scores_df2.max_depth, scores_df2.train_acc_rf, label='train', marker='o')
    plt.plot(scores_df2.max_depth, scores_df2.validate_acc_rf, label='validate', marker='o')
    plt.axhline(baseline_accuracy, linewidth=2, color='black', label='baseline')
    plt.fill_between(scores_df2.max_depth, scores_df2.train_acc_rf, scores_df2.validate_acc_rf, alpha=.4)
    plt.xlabel('Max Depth for Random Forest')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(1,11, step=1))
    plt.title('Where do I begin to overfit?')
    plt.legend()
    plt.show()
    
        # plot feature importance
    plt.figure(figsize=(12,12))
    plt.bar(X_train.columns, important_features)
    plt.title(f"Feature Importance")
    plt.xlabel(f"Features")
    plt.ylabel(f"Importance")
    plt.xticks(rotation = 60)
    plt.show()
    
    # KNN
    print(f"KNN")
    print(f"The number of features sent in : {len(X_train.columns)} and are {X_train.columns.tolist()}.")

    # run for loop and plot
    metrics = []
    for k in range(1, nn):
        
        # make the model
        knn = KNeighborsClassifier(n_neighbors=k, weights='uniform')
        
        # fit the model
        knn.fit(X_train, y_train)
        
        # calculate accuracy
        train_score = knn.score(X_train, y_train)
        validate_score = knn.score(X_validate, y_validate)
        
        # append to df metrics
        metrics.append([k, train_score, validate_score])

        # turn to df
        metrics_df = pd.DataFrame(metrics, columns=['k', 'train score', 'validate score'])
      
        # make new column
        metrics_df['difference'] = metrics_df['train score'] - metrics_df['validate score']
    min_diff_idx = np.abs(metrics_df['difference']).argmin()
    n = metrics_df.loc[min_diff_idx, 'k']

    # make plottable df without the difference column
    metrics_plot = metrics_df.drop(columns='difference')
    print(f"{n} is the number of neighbors that produces the best fit model.")
    print(f"The accuracy score for the train model is {round(train_score,2)}.")
    print(f"The accuracy score for the validate model is {round(validate_score,2)}.")
    
    
    # plot the data
    metrics_plot.set_index('k').plot(figsize = (14,12))
    plt.axvline(x=n, color='black', linestyle='--', linewidth=1, label='best fit neighbor size')
    plt.axhline(y=train_score, color='blue', linestyle='--', linewidth=1, label='train accuracy')
    plt.axhline(y=validate_score, color='orange', linestyle='--', linewidth=1, label='validate accuracy')
    plt.fill_between(metrics_df.k, metrics_df['train score'], metrics_df['validate score'], alpha=.4)
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0,nn,1))
    plt.legend()
    plt.grid()
    plt.show()
    
    
    # LOGISTIC REGRESSION TRAIN
    print(f"LOGISTIC REGRESSION")
    print(f"Train Dataset")
    #create it
    logit = LogisticRegression(random_state=123)

    #fit it
    logit.fit(X_train, y_train)

    #use it
    lt_score = logit.score(X_train, y_train)
    print(f"The train model's accuracy is {round(lt_score,2)}")
    
    #baseline
    print(f"The baseline accuracy is {round(baseline_accuracy,2)}.") 
    
    #classification report
    print(classification_report(y_train, logit.predict(X_train)))

    # LOGISTIC REGRESSION VALIDATE
    print(f"LOGISTIC REGRESSION")
    print(f"Validate Dataset")
    #create it
    logit2 = LogisticRegression(random_state=123)

    #fit it
    logit2.fit(X_validate, y_validate)

    #use it
    lt_score2 = logit2.score(X_validate, y_validate)
    print(f"The validate model's accuracy is {round(lt_score2,2)}")

    #baseline
    print(f"The baseline accuracy is {round(baseline_accuracy,2)}.")
    
    #classification report
    print(classification_report(y_validate, logit2.predict(X_validate)))