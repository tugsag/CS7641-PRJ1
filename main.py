from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split, learning_curve, validation_curve, StratifiedKFold, GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, plot_confusion_matrix, f1_score, plot_roc_curve, roc_auc_score
from itertools import product
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

SEED = 903454028
np.random.seed(SEED)


def plot_lc(estimator, title, X, y, axes=None, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), path=None):
    plt.title(title)
    if ylim is not None:
        plt.set_ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # plot learning curve
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    plt.legend(loc="best")

    plt.savefig(path)

def plot_vc(estimator, X, y, title, param_name, xlabel, param_range, scoring, cv, path=None):
    train_score, test_score = validation_curve(estimator, X, y, param_name=param_name, param_range=param_range, cv=cv, scoring=scoring, n_jobs=8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Score')
    train_score_mean = train_score.mean(axis=1)
    test_score_mean = test_score.mean(axis=1)
    plt.semilogx(param_range, train_score_mean, label='Training Score', color='darkorange')
    plt.semilogx(param_range, test_score_mean, label='CV Score', color='navy')
    plt.legend(loc='best')
    plt.savefig(path)

###### MODELS ######

def dt(d, id=None):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, shuffle=True, random_state=SEED)
    split = StratifiedShuffleSplit(n_splits=1, test_size=.33, random_state=SEED)
    for i, j in split.split(d, d['y']):
        train_set = d.loc[i]
        test_set = d.loc[j]
    y_train, y_test = train_set['y'], test_set['y']
    X_train, X_test = train_set.drop('y', axis=1), test_set.drop('y', axis=1)
    if id=='E':
        scoring = 'f1'
    else:
        scoring = 'roc_auc'

    model_naive = DecisionTreeClassifier()
    model_naive.fit(X_train, y_train)
    pred = model_naive.predict(X_test)
    cv = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
    print('Accuracy of default model is: ', accuracy_score(y_test, pred))
    print('F1 of default model is: ', f1_score(y_test, pred))
    print('roc_auc of default is: ', roc_auc_score(y_test, pred))
    plot_lc(model_naive, 'Learning Curve - Default Model', X_train, y_train, cv=cv, n_jobs=-1, path='figures/DT_{}_Default_learning.png'.format(id))
    plt.clf()
    print('plot saved')

    model = DecisionTreeClassifier(random_state=SEED)
    criterion = ['gini', 'entropy']
    max_depth = np.arange(1, 50)
    ccp_alpha = [.005, .003, .002, .001]
    grid = dict(criterion=criterion, max_depth=max_depth, ccp_alpha=ccp_alpha)
    # cv = KFold(n_splits=3, random_state=SEED, shuffle=True)
    out = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring=scoring, error_score=0)
    result = out.fit(X_train, y_train)
    best_model = result.best_estimator_
    best_params = result.best_params_

    print("Best {} {} using params {}".format(scoring, result.best_score_, best_params))

    # ccpAlphas
    path = model.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas[:-1]


    # learning_curve
    plot_lc(best_model, 'Learning Curve - Best Model', X_train, y_train, cv=cv, n_jobs=-1, path='figures/DT_{}_Best_learning.png'.format(id))
    plt.clf()

    # validation curve
    param_range = np.linspace(ccp_alphas[0], ccp_alphas[-1], num=20)
    plot_vc(best_model, X_train, y_train, 'Validation Curve', 'ccp_alpha', 'ccp_alphas', param_range, scoring, cv=cv, path='figures/DT_{}_Best_valid.png'.format(id))
    plt.clf()

    plot_roc_curve(best_model, X_test, y_test)
    plt.savefig('figures/DT_{}_ROC.png'.format(id))
    plt.clf()

    best_model.fit(X_train, y_train)
    pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    c_matrix = confusion_matrix(y_test, pred)
    c_report = classification_report(y_test, pred)

    print('Best Accuracy: ', accuracy)
    print('confusion_matrix: \n', c_matrix)
    print('classification_report: \n', c_report)

    plot_confusion_matrix(best_model, X_test, y_test, normalize='pred')
    plt.savefig("figures/DT_{}_Confusion.png".format(id))
    plt.clf()


def svm(d, id=None):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=SEED)
    split = StratifiedShuffleSplit(n_splits=1, test_size=.33, random_state=SEED)
    for i, j in split.split(d, d['y']):
        train_set = d.loc[i]
        test_set = d.loc[j]
    y_train, y_test = train_set['y'], test_set['y']
    X_train, X_test = train_set.drop('y', axis=1), test_set.drop('y', axis=1)

    if id == 'E':
        scoring = 'f1'
    else:
        scoring = 'roc_auc'

    model_naive = SVC()
    model_naive.fit(X_train, y_train)
    pred = model_naive.predict(X_test)
    cv = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
    print('Accuracy of default model is: ', accuracy_score(y_test, pred))
    print('F1 of default model is: ', f1_score(y_test, pred))
    print('roc_auc of default is: ', roc_auc_score(y_test, pred))
    plot_lc(model_naive, 'Learning Curve - Default Model', X_train, y_train, cv=cv, n_jobs=-1, path='figures/SVM_{}_Default_learning.png'.format(id))
    plt.clf()
    print('plot saved')

    model = SVC(random_state=SEED)
    kernel = ['poly', 'rbf', 'sigmoid']
    C = [.001, .01, .1, 1, 10, 100, 1000]
    gamma = ['scale']
    grid = dict(kernel=kernel, C=C, gamma=gamma)
    out = GridSearchCV(estimator=model, param_grid=grid, cv=cv, scoring='f1', n_jobs=8, error_score=0)
    result = out.fit(X_train, y_train)
    best_model = result.best_estimator_
    best_params = result.best_params_

    print("Best {} {} using params {}".format(scoring, result.best_score_, result.best_params_))

    # learning curve
    plot_lc(best_model,'Learning Curve - Best Model', X_train, y_train, cv=cv, n_jobs=-1, path='figures/SVM_{}_Best_learning.png'.format(id))
    plt.clf()

    # validation_curve
    param_range = np.logspace(-3, 3, 10)
    plot_vc(best_model, X_train, y_train, 'Validation Curve', 'C', 'C', param_range, scoring, cv=cv, path='figures/SVM_{}_Best_valid.png'.format(id))
    plt.clf()

    plot_roc_curve(best_model, X_test, y_test)
    plt.savefig('figures/SVM_{}_ROC.png'.format(id))
    plt.clf()

    # C = 1
    # model_1 = SVC(C=5)
    # model_1.fit(X_train, y_train)
    # pred = model_1.predict(X_test)
    # plot_lc(model_1, 'Learning Curve - C = 5', X_train, y_train, cv=cv, n_jobs=-1, path='figures/SVM_E_model5_learning.png')
    # plt.clf()

    best_model.fit(X_train, y_train)
    pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    c_matrix = confusion_matrix(y_test, pred)
    c_report = classification_report(y_test, pred)

    print('Accuracy: ', accuracy)
    print('confusion_matrix: \n', c_matrix)
    print('classification_report: \n', c_report)

    plot_confusion_matrix(best_model, X_test, y_test, normalize='pred')
    plt.savefig('figures/SVM_{}_Confusion.png'.format(id))
    plt.clf()

def knn(d, id=None):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=SEED)
    split = StratifiedShuffleSplit(n_splits=1, test_size=.33, random_state=SEED)
    for i, j in split.split(d, d['y']):
        train_set = d.loc[i]
        test_set = d.loc[j]
    y_train, y_test = train_set['y'], test_set['y']
    X_train, X_test = train_set.drop('y', axis=1), test_set.drop('y', axis=1)

    if id == 'E':
        scoring = 'f1'
    else:
        scoring = 'roc_auc'

    model_naive = KNeighborsClassifier()
    model_naive.fit(X_train, y_train)
    pred = model_naive.predict(X_test)
    cv = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
    print('Accuracy of default model is: ', accuracy_score(y_test, pred))
    print('F1 of default model is: ', f1_score(y_test, pred))
    print('roc_auc of default is: ', roc_auc_score(y_test, pred))
    plot_lc(model_naive, 'Learning Curve - Default Model', X_train, y_train, cv=cv, n_jobs=-1, path='figures/KNN_{}_Default_learning.png'.format(id))
    plt.clf()
    print('plot saved')


    model = KNeighborsClassifier()
    n_neighbors = range(1, 21, 2)
    weights = ['uniform', 'distance']
    metric = ['euclidean', 'manhattan', 'minkowski']
    grid = dict(n_neighbors=n_neighbors, weights=weights, metric=metric)
    out = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring=scoring, error_score=0)
    result = out.fit(X_train, y_train)
    best_model = result.best_estimator_
    best_params = result.best_params_

    print("Best {} {} using params {}".format(scoring, result.best_score_, result.best_params_))

    # learning curve
    plot_lc(best_model, 'Learning Curve - Best Model', X_train, y_train, cv=cv, n_jobs=-1, path='figures/KNN_{}_Best_learning.png'.format(id))
    plt.clf()

    # validation_curve
    param_range = np.arange(1, 21)
    plot_vc(model_naive, X_train, y_train, 'Validation Curve', 'n_neighbors', '# Neighbors', param_range, scoring, cv=cv, path='figures/KNN_{}_Best_valid.png'.format(id))
    plt.clf()

    plot_roc_curve(best_model, X_test, y_test)
    plt.savefig('figures/KNN_{}_ROC.png'.format(id))
    plt.clf()

    best_model.fit(X_train, y_train)
    pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    c_matrix = confusion_matrix(y_test, pred)
    c_report = classification_report(y_test, pred)

    print('Accuracy: ', accuracy)
    print('confusion_matrix: \n', c_matrix)
    print('classification_report: \n', c_report)

    plot_confusion_matrix(best_model, X_test, y_test, normalize='pred')
    plt.savefig('figures/KNN_{}_Confusion.png'.format(id))
    plt.clf()


def gb(d, id=None):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=SEED)
    split = StratifiedShuffleSplit(n_splits=1, test_size=.33, random_state=SEED)
    for i, j in split.split(d, d['y']):
        train_set = d.loc[i]
        test_set = d.loc[j]
    y_train, y_test = train_set['y'], test_set['y']
    X_train, X_test = train_set.drop('y', axis=1), test_set.drop('y', axis=1)

    if id == 'E':
        scoring = 'f1'
    else:
        scoring = 'roc_auc'

    model_naive = GradientBoostingClassifier()
    model_naive.fit(X_train, y_train)
    pred = model_naive.predict(X_test)
    cv = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
    print('Accuracy of default model is: ', accuracy_score(y_test, pred))
    print('F1 of default model is: ', f1_score(y_test, pred))
    print('roc_auc of default is: ', roc_auc_score(y_test, pred))
    plot_lc(model_naive, 'Learning Curve - Default Model', X_train, y_train, cv=cv, n_jobs=12, path='figures/GB_{}_Default_learning.png'.format(id))
    plt.clf()
    print('plot saved')

    model = GradientBoostingClassifier(random_state=SEED)
    n_estimators = [50, 100, 200, 300, 500]
    max_depth = np.arange(1, 11, 2)
    learning_rate = [.01, .1, .5]
    grid = dict(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
    # cv = KFold(n_splits=3, random_state=SEED, shuffle=True)
    out = GridSearchCV(estimator=model, param_grid=grid, n_jobs=16, cv=cv, scoring=scoring, error_score=0, verbose=5)
    result = out.fit(X_train, y_train)
    best_model = result.best_estimator_
    best_params = result.best_params_

    print("Best {} {} using params {}".format(scoring, result.best_score_, result.best_params_))
    # best_model = GradientBoostingClassifier(ccp_alpha=.001, learning_rate=.1, n_estimators=50, random_state=SEED)

    # learning curve
    plot_lc(best_model, 'Learning Cuve - Best Model', X_train, y_train, cv=cv, n_jobs=12, path='figures/GB_{}_Best_learning.png'.format(id))
    plt.clf()

    # validation_curve
    param_range = max_depth
    plot_vc(best_model, X_train, y_train, 'Validation Curve', 'max_depth', 'Max Depth', param_range, scoring, cv=cv, path='figures/GB_{}_Best_valid.png'.format(id))
    plt.clf()

    plot_roc_curve(best_model, X_test, y_test)
    plt.savefig('figures/GB_{}_ROC.png'.format(id))
    plt.clf()

    best_model.fit(X_train, y_train)
    pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    c_matrix = confusion_matrix(y_test, pred)
    c_report = classification_report(y_test, pred)

    print('Accuracy: ', accuracy)
    print('confusion_matrix: \n', c_matrix)
    print('classification_report: \n', c_report)

    plot_confusion_matrix(best_model, X_test, y_test, normalize='pred')
    plt.savefig('figures/GB_{}_Confusion.png'.format(id))
    plt.clf()



def nn(d, id=None):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=SEED)
    split = StratifiedShuffleSplit(n_splits=1, test_size=.33, random_state=SEED)
    for i, j in split.split(d, d['y']):
        train_set = d.loc[i]
        test_set = d.loc[j]
    y_train, y_test = train_set['y'], test_set['y']
    X_train, X_test = train_set.drop('y', axis=1), test_set.drop('y', axis=1)

    if id == 'E':
        scoring = 'f1'
    else:
        scoring = 'roc_auc'

    model_naive = MLPClassifier()
    model_naive.fit(X_train, y_train)
    pred = model_naive.predict(X_test)
    cv = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
    print('Accuracy of default model is: ', accuracy_score(y_test, pred))
    print('F1 of default model is: ', f1_score(y_test, pred))
    print('roc_auc of default is: ', roc_auc_score(y_test, pred))
    plot_lc(model_naive, 'Learning Curve - Default Model', X_train, y_train, cv=cv, n_jobs=8, path='figures/ANN_{}_Default_learning.png'.format(id))
    plt.clf()
    print('plot saved')

    model = MLPClassifier(random_state=SEED, max_iter=500)
    hidden_layer_sizes = [(5,), (5,5,), (10, 10), (7,), (10,)]
    activation = ['logistic', 'tanh', 'relu']
    alpha = [.0001, .001, .01, .1, .2]
    grid = dict(hidden_layer_sizes=hidden_layer_sizes, activation=activation, alpha=alpha)
    # cv = KFold(n_splits=3, random_state=SEED, shuffle=True)
    out = GridSearchCV(estimator=model, param_grid=grid, n_jobs=8, cv=cv, scoring=scoring, error_score=0)
    result = out.fit(X_train, y_train)
    best_model = result.best_estimator_
    best_params = result.best_params_

    print("Best {} {} using params {}".format(scoring, result.best_score_, result.best_params_))

    # learning curve
    plot_lc(best_model, 'Learning Cuve - Best Model', X_train, y_train, cv=cv, n_jobs=8, path='figures/ANN_{}_Best_learning.png'.format(id))
    plt.clf()

    # validation_curve
    param_range = np.linspace(.0001, .2, 10)
    plot_vc(best_model, X_train, y_train, 'Validation Curve', 'alpha', 'Alphas', param_range, scoring, cv=cv, path='figures/ANN_{}_Best_valid.png'.format(id))
    plt.clf()

    plot_roc_curve(best_model, X_test, y_test)
    plt.savefig('figures/ANN_{}_ROC.png'.format(id))
    plt.clf()

    best_model.fit(X_train, y_train)
    pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    c_matrix = confusion_matrix(y_test, pred)
    c_report = classification_report(y_test, pred)

    print('Accuracy: ', accuracy)
    print('confusion_matrix: \n', c_matrix)
    print('classification_report: \n', c_report)

    plot_confusion_matrix(best_model, X_test, y_test, normalize='pred')
    plt.savefig('figures/ANN_{}_Confusion.png'.format(id))
    plt.clf()


if __name__ == "__main__":
    d1 = pd.read_csv('epilepsy/epilepsy.csv')
    # d2 = pd.read_csv('pulsar/pulsar_stars.csv')
    d3 = pd.read_csv('gamma/gamma.csv')

    # preprocess epilepsy data
    temp = np.array(d1['y'].values.tolist())
    d1['y'] = np.where(temp > 1, 0, temp).tolist()
    d1 = d1.drop('Unnamed', axis=1)

    # preprocess pulsar data
    # d2 = d2.rename(columns={'target_class': 'y'})

    # # preprocess gamma data
    d3 = d3.rename(columns={'class': 'y'})
    d3 = d3.drop(d3.columns[0], axis=1)
    m = {'g': 1, 'h': 0}
    d3['y'] = d3['y'].map(m)


    x = input('''Choose algorithm:
                    Decision tree: DT
                    SVM: SVM
                    Boosting: B
                    KNN: KNN
                    ANN: ANN ''')
    y = input('''Choose dataset:
                    epilepsy: E
                    gamma: G ''')

    if x == 'DT':
        if y == 'E':
            dt(d1, id=y)
        elif y == 'G':
            dt(d3, id=y)
        else:
            dt(d2, id=y)
    elif x == 'SVM':
        if y == 'E':
            svm(d1, id=y)
        elif y == 'G':
            svm(d3, id=y)
        else:
            svm(d2, id=y)
    elif x == 'B':
        if y == 'E':
            gb(d1, id=y)
        elif y == 'G':
            gb(d3, id=y)
        else:
            gb(d2, id=y)
    elif x == 'KNN':
        if y == 'E':
            knn(d1, id=y)
        elif y == 'G':
            knn(d3, id=y)
        else:
            knn(d2, id=y)
    elif x == 'ANN':
        if y == 'E':
            nn(d1, id=y)
        elif y == 'G':
            nn(d3, id=y)
        else:
            nn(d2, id=y)
    else:
        print('invalid entry')
