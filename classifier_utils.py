import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
#from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score, roc_curve, confusion_matrix
from sklearn.model_selection import KFold
import matplotlib
import matplotlib.pyplot as plt


def auroc_aupr(ytrue, ypred):
    return roc_auc_score(ytrue, ypred), average_precision_score(ytrue, ypred)


def run_tests(tests, xtrain, ytrain, xtest, ytest, respath, save_file=None, retrain=False):
    result = pd.DataFrame(columns=['test_type', 'name', 'value'])
    feature_imp = pd.DataFrame(columns=['test_type', 'feature', 'imp', 'rank'])
    kf = KFold(n_splits=5)
    for test in tests:
        print(test)
        model, random_grid = get_model_grid(test)
        scoring = 'roc_auc'
        n_iter = 5 if test == "ebm" else 25
        if retrain:
            random_search = RandomizedSearchCV(estimator=model, param_distributions=random_grid, verbose=1,
                                               scoring=scoring, n_iter=n_iter, cv=3, random_state=1, n_jobs=-1)
            random_search.fit(xtrain, ytrain)
            model = random_search.best_estimator_
            if save_file:
                pickle.dump(model, open(respath + '/trained_model/{}_{}.p'.format(save_file, test), 'wb'))
        else:
            model = pickle.load(open(respath + '/trained_model/{}_{}.p'.format(save_file, test), 'rb'))

        cv_perfs = pd.DataFrame(columns=["name", "value"])
        for train_index, _ in kf.split(xtrain):
            xtrain_cv = xtrain.iloc[train_index]
            ytrain_cv = ytrain.iloc[train_index]
            model.fit(xtrain_cv, ytrain_cv)

            train_auroc, train_aupr = auroc_aupr(ytrain_cv, model.predict_proba(xtrain_cv)[:, 1])
            test_auroc, test_aupr = auroc_aupr(ytest, model.predict_proba(xtest)[:, 1])
            #cv_perfs = cv_perfs.append({'name': 'auroc_train', 'value': train_auroc}, ignore_index=True, sort=False)
            cv_perfs.loc[len(cv_perfs)] = {'name': 'auroc_train', 'value': train_auroc}
            
            #cv_perfs = cv_perfs.append({'name': 'aupr_train', 'value': train_aupr}, ignore_index=True, sort=False)
            cv_perfs.loc[len(cv_perfs)] = {'name': 'aupr_train', 'value': train_aupr}
            
            #cv_perfs = cv_perfs.append({'name': 'auroc_test', 'value': test_auroc}, ignore_index=True, sort=False)
            cv_perfs.loc[len(cv_perfs)] = {'name': 'auroc_test', 'value': test_auroc}
            
            #cv_perfs = cv_perfs.append({'name': 'aupr_test', 'value': test_aupr}, ignore_index=True, sort=False)
            cv_perfs.loc[len(cv_perfs)] = {'name': 'aupr_test', 'value': test_aupr}
            
            score_dict = get_score_dict(model, list(xtrain_cv), xtrain_cv, ytrain_cv)
            if len(score_dict) > 0:
                cv_imps = imp_to_rank(score_dict)
            else:
                cv_imps = pd.DataFrame()

            cv_imps['test_type'] = test
            #feature_imp = feature_imp.append(cv_imps, ignore_index=True, sort=False)
            feature_imp = pd.concat([feature_imp, cv_imps], ignore_index=True)
            
        cv_perfs['test_type'] = test
       # result = result.append(cv_perfs, ignore_index=True, sort=False)
        result = pd.concat([result, cv_perfs], ignore_index=True)

    if save_file:
        result.to_csv(respath + '/trained_model/{}_perf.csv'.format(save_file))
        feature_imp.to_csv(respath + '/trained_model/{}_featureimp.p'.format(save_file))
    return result, feature_imp


def get_model_grid(test):
    if test == "xgb":
        model = XGBClassifier()
        random_grid = {'learning_rate': [10 ** (-1 * x) for x in np.linspace(start=0, stop=5, num=10)],
                       'n_estimators': [int(x) for x in np.linspace(start=10, stop=200, num=20)],
                       'max_depth': [1, 3, 5, 7, 10],
                       'reg_alpha': [1, 2, 3],
                       'reg_lambda': [2, 3, 5],
                       'gamma': [0, 0.1, 0.2],
                       'booster': ['gbtree', 'gblinear', 'dart']
                       }
    elif test == "gb":
        model = GradientBoostingClassifier()
        random_grid = {'learning_rate': [10 ** (-1 * x) for x in np.linspace(start=0, stop=5, num=10)],
                       'n_estimators': [int(x) for x in np.linspace(start=50, stop=1000, num=50)],
                       'max_features': ['auto', 'log2', None],
                       'max_depth': [int(x) for x in np.linspace(3, 10, num=5)],
                       'min_samples_split': [2, 3, 5, 7, 10],
                       'min_samples_leaf': [1, 2, 4]}
    # elif test == 'ebm':
    #     model = ExplainableBoostingClassifier()
    #     random_grid = {'learning_rate': [10 ** (-1 * x) for x in np.linspace(start=0, stop=5, num=10)],
    #                    'early_stopping_tolerance': [10 ** (-1 * x) for x in
    #                                                 np.linspace(start=3, stop=7, num=5)]
    #                    } #'training_step_episodes': [1, 2, 3], 'n_estimators': [int(x) for x in np.linspace(start=8, stop=100, num=8)],'max_tree_splits': [2, 3, 5]
        
    elif test == "rf":
        model = RandomForestClassifier()  # class_weight='balanced')
        max_depth = [int(x) for x in np.linspace(1, 110, num=5)]
        max_depth.append(None)
        random_grid = {'n_estimators': [int(x) for x in np.linspace(start=10, stop=2000, num=20)],
                       'max_features': ['auto', 'log2', None],
                       'max_depth': max_depth,
                       'min_samples_split': [2, 3, 5, 7, 10],
                       'min_samples_leaf': [1, 2, 4],
                       'bootstrap': [True, False]}
    elif test == "nn":
        model = MLPClassifier()
        random_grid = {'max_iter': [100, 1000, 2000],
                       'hidden_layer_sizes': [(20,), (50,), (100,), (200,), (1000,)],
                       'activation': ['logistic', 'tanh', 'relu'],
                       'solver': ['lbfgs', 'sgd', 'adam'],
                       'alpha': [10 ** (-1 * x) for x in np.linspace(start=0, stop=5, num=5)],
                       'learning_rate': ['constant', 'invscaling', 'adaptive'],
                       'learning_rate_init': [10 ** (-1 * x) for x in np.linspace(start=0, stop=5, num=20)],
                       'momentum': [0.9, 0.99],
                       'early_stopping': [False, True],
                       'n_iter_no_change': [10, 100],
                       'epsilon': [10 ** (-1 * x) for x in np.linspace(start=5, stop=10, num=5)]
                       }
    elif test == 'l1':
        model = LogisticRegression(penalty='l1', solver='liblinear')
        random_grid = {'C': [10.**log_c for log_c in np.arange(-3, 4, 0.1)]}
    else:
        print("{} not available".format(test))
        return None, None

    return model, random_grid


def get_score_dict(model, features, xtrain=None, ytrain=None):
    if isinstance(model, XGBClassifier):
        if model.booster == 'gblinear':
            gain = model.get_booster().get_score(importance_type='weight')
        else:
            gain = model.get_booster().get_score(importance_type='gain')
        
        # normalize
        sum_gain = np.sum(list(gain.values()))
        if sum_gain == 0:
            sum_gain = 1 # making sure we don't get all nans, just all zeros
        return {f: (gain[f]/sum_gain if f in gain else 0) for f in features}
    elif isinstance(model, GradientBoostingClassifier) or isinstance(model, RandomForestClassifier):
        imp = model.feature_importances_
        return {features[i]: imp[i] for i in range(len(imp))}
    # elif isinstance(model, ExplainableBoostingClassifier):
    #     glob = model.explain_global()._internal_obj["overall"]
    #     return {glob["names"][i]: glob["scores"][i] for i in range(len(glob["names"]))}
    elif isinstance(model, LogisticRegression):
        freq_features = np.zeros(len(features))
        log_cs = np.arange(-3, 4, 0.5)
        for log_c in log_cs:
            model.C = 10.**log_c
            model.fit(xtrain, ytrain)
            freq_features[np.nonzero(model.coef_)[1]] += 1
        return {features[i]: freq_features[i] for i in range(len(freq_features))}
    else:
        print("Feature importance for this test is not available.")
        return {}


def imp_to_rank(score_dict):
    ''''' Rank as if importance is higher, the better.
    A score dict is a dictionary where the keys are feature names and their values are their importances '''''
    ranks = pd.DataFrame(columns=['feature', 'imp', 'rank'])
    count = {col: 0 for col in score_dict.keys()}
    for col in score_dict:
        count[col] += abs(score_dict[col])

    i = 0
    for col in sorted(count, key=count.get, reverse=True):
        #ranks = ranks.append({'feature': col, 'imp': score_dict[col], 'rank': i}, ignore_index=True)
        ranks.loc[len(ranks)] = {'feature': col, 'imp': score_dict[col], 'rank': i}
        i += 1
    return ranks


def train_per_feature(tests, xtrain, ytrain, xtest, ytest, respath, save_file):
    # Load feature importance from previous run and get ranks from the means
    feature_imp = pd.read_csv(respath + '/trained_model/{}_featureimp.p'.format(save_file))
    grouped_imp = feature_imp.groupby(['test_type', 'feature']).agg({'imp': 'mean'})
    grouped_imp.reset_index(inplace=True)

    ranks = pd.DataFrame(columns=['test', 'feature', 'imp', 'rank'])
    result = pd.DataFrame(columns=['test_type', 'num_features', 'name', 'value'])
    kf = KFold(n_splits=5)

    for test in tests:
        cur_rank = imp_to_rank(grouped_imp[grouped_imp.test_type == test].set_index('feature')['imp'].to_dict())
        cur_rank['test'] = test
        #ranks = ranks.append(cur_rank, ignore_index=True, sort=False)
        ranks = pd.concat([ranks, cur_rank], ignore_index=True)
        
    
        model = pickle.load(open(respath + '/trained_model/{}_{}.p'.format(save_file, test), 'rb'))
        features = []
        for i in range(cur_rank.shape[0]):
            feature_added = cur_rank[cur_rank['rank'] == i]['feature'].iloc[0]
            features.append(feature_added)

            if test == 'ebm':
                model.feature_names = None
                model.feature_types = None

            cv_perfs = pd.DataFrame(columns=["name", "value"])
            itr = 0
            for train_index, _ in kf.split(xtrain):
                xtrain_cv = xtrain.iloc[train_index][features]
                ytrain_cv = ytrain.iloc[train_index]
                model.fit(xtrain_cv, ytrain_cv)

                train_auroc, train_aupr = auroc_aupr(ytrain_cv, model.predict_proba(xtrain_cv)[:, 1])
                test_auroc, test_aupr = auroc_aupr(ytest, model.predict_proba(xtest[features])[:, 1])
                
                #cv_perfs = cv_perfs.append({'name': 'auroc_train', 'value': train_auroc}, ignore_index=True, sort=False)
                cv_perfs.loc[len(cv_perfs)] = {'name': 'auroc_train', 'value': train_auroc}
                
                #cv_perfs = cv_perfs.append({'name': 'aupr_train', 'value': train_aupr}, ignore_index=True, sort=False)
                cv_perfs.loc[len(cv_perfs)] = {'name': 'aupr_train', 'value': train_aupr}
                
                #cv_perfs = cv_perfs.append({'name': 'auroc_test', 'value': test_auroc}, ignore_index=True, sort=False)
                cv_perfs.loc[len(cv_perfs)] = {'name': 'auroc_test', 'value': test_auroc}
                
                #cv_perfs = cv_perfs.append({'name': 'aupr_test', 'value': test_aupr}, ignore_index=True, sort=False)
                cv_perfs.loc[len(cv_perfs)] = {'name': 'aupr_test', 'value': test_aupr}
                
                
                
                
                cv_perfs.loc[len(cv_perfs)] = {'name': 'auroc_train', 'value': train_auroc, 'itr': itr}
                cv_perfs.loc[len(cv_perfs)] = {'name': 'aupr_train', 'value': train_aupr, 'itr': itr}
                cv_perfs.loc[len(cv_perfs)] = {'name': 'auroc_test', 'value': test_auroc, 'itr': itr}
                cv_perfs.loc[len(cv_perfs)] = {'name': 'aupr_test', 'value': test_aupr, 'itr': itr}

                precision, recall, threshold = recall_from_best_f1(ytest, model.predict_proba(xtest[features])[:, 1])
                cv_perfs.loc[len(cv_perfs)] = {'name': 'precision', 'value': precision, 'itr': itr}
                cv_perfs.loc[len(cv_perfs)] = {'name': 'recall', 'value': recall, 'itr': itr}
                cv_perfs.loc[len(cv_perfs)] = {'name': 'threshold', 'value': threshold, 'itr': itr}
                itr += 1
            cv_perfs['test_type'] = test
            cv_perfs['num_features'] = i + 1
            
            result = pd.concat([result, cv_perfs], ignore_index=True, sort=False)

    ranks.to_csv(respath + '/trained_model/{}_overallranks.csv'.format(save_file))
    result.to_csv(respath + '/trained_model/{}_perfperfeature.csv'.format(save_file))
    return result, ranks


def recall_from_best_f1(ytrue, ypred):
    precision, recall, thresholds = precision_recall_curve(ytrue, ypred)
    f1s = [f1_score(ytrue, np.array(ypred >= t).astype(int)) for t in thresholds]
    i = np.argmax(f1s)
    return precision[i], recall[i], thresholds[i]


def get_auroc_aupr(tests, xtrain, ytrain, xtest, ytest, respath, save_file, nfeatures):
    overall_ranks = pd.read_csv(respath + '/trained_model/{}_overallranks.csv'.format(save_file))
    result_auc = pd.DataFrame(columns=['test_type', 'num_features', 'name', 'value'])
    result_thresholds = pd.DataFrame(columns=['test_type', 'num_features',  'tpr', 'fpr',
                                    'precision', 'threshold'])
    # Initiate for AUROC and AUPR plots
    fig_counter = 0
    for i in nfeatures:
        plt.figure(fig_counter, figsize=(10, 10))
        plt.figure(fig_counter + 1, figsize=(10, 10))
        labels = {'auroc': [], 'aupr': []}
        for test in tests:
            cur_rank = overall_ranks[overall_ranks.test == test]
            model = pickle.load(open(respath + '/trained_model/{}_{}.p'.format(save_file, test), 'rb'))
            features = cur_rank[cur_rank['rank'] <= i]['feature'].values

            if test == 'ebm':
                model.feature_names = None
                model.feature_types = None
            elif test == 'l1':
                model.multi_class = 'auto'
            elif test == 'gb':
                model.ccp_alpha = 0
            elif test == 'rf':
                model.ccp_alpha = 0
                model.max_samples = None

            cur_auc = pd.DataFrame(columns=["name", "value"])

            model.fit(xtrain[features], ytrain)

            train_auroc, train_aupr = auroc_aupr(ytrain, model.predict_proba(xtrain[features])[:, 1])
            ytest_pred = model.predict_proba(xtest[features])[:, 1]
            test_auroc, test_aupr = auroc_aupr(ytest, ytest_pred)
            
            cur_auc.loc[len(cur_auc)] = {'name': 'auroc_train', 'value': train_auroc}
            cur_auc.loc[len(cur_auc)] = {'name': 'auroc_test', 'value': test_auroc}
            cur_auc.loc[len(cur_auc)] = {'name': 'test_aupr', 'value': test_aupr}
            
            
            
            
            cur_auc['test_type'] = test
            cur_auc['num_features'] = i + 1

            labels['auroc'].append('{} (auroc: {:.2f})'.format(test, test_auroc))
            
            plot_auroc(fig_counter, ytest, ytest_pred, labels['auroc'], title="ROC with {} features".format(i))

            labels['aupr'].append('{} (ap: {:.2f})'.format(test, test_aupr))
            plot_aupr(fig_counter + 1, ytest, ytest_pred, labels['aupr'], title="PR with {} features".format(i))

            
            result_auc = pd.concat([result_auc, cur_auc], ignore_index=True)
            
            perfs = get_perf_thresholds(ytest, ytest_pred)
            perfs['test_type'] = test
            perfs['num_features'] = i + 1
            result_thresholds = pd.concat([result_thresholds, perfs], ignore_index=True)
        fig_counter += 2

    result_auc.to_csv(respath + '/trained_model/{}_somefeature.csv'.format(save_file))
    result_thresholds.to_csv(respath + '/trained_model/{}_somefeature_thresholds.csv'.format(save_file))
    return result_auc


def plot_auroc(fig_counter, ytrue, ypred, labels, title="ROC"):
    try:
        fpr, tpr, thresholds = roc_curve(ytrue, ypred)
        plt.figure(fig_counter)
        fig_counter += 1
        plt.plot(fpr, tpr)
        plt.title(title)
        plt.xlabel('False Positive Rate (FP/N)')
        plt.ylabel('Sensitivity/Recall (TP/P)')
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.xticks(np.arange(0, 1.05, 0.1))
        plt.yticks(np.arange(0, 1.05, 0.1))
        plt.legend(labels=labels)
    except ValueError as e:
        print(e)
    return fig_counter


def plot_aupr(fig_counter, ytrue, ypred, labels, title="PRC"):
    try:
        precision, recall, thresholds = precision_recall_curve(ytrue, ypred)
        plt.figure(fig_counter)
        fig_counter += 1
        plt.plot(recall, precision)
        plt.title(title)
        plt.xlabel('Recall (TP/P)')
        plt.ylabel('Precision (TP/(TP + FP))')
        plt.legend(labels=labels)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.xticks(np.arange(0, 1.05, 0.1))
        plt.yticks(np.arange(0, 1.05, 0.1))
    except ValueError as e:
        print(e)
    return fig_counter


def get_perf_thresholds(ytrue, ypred):
    try:
        fpr, tpr, thresholds = roc_curve(ytrue, ypred)
        precision = []
        for t in thresholds:
            tn, fp, fn, tp = confusion_matrix(ytrue, (ypred >= t).astype(int)).ravel()
            if (tp + fp) > 0:
                precision.append(tp/(tp + fp))
            else:
                precision.append(np.nan)
    except ValueError as e:
        print(e)
    out = pd.DataFrame()
    out['fpr'] = fpr
    out['tpr'] = tpr
    out['threshold'] = thresholds
    out['precision'] = precision
    return out
