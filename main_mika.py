import utils
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
import classifier_utils
import os
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from c_stg.utils import import_per_data_type
from c_stg.data_processing import DataContainer
from c_stg.params import Cstg_params
import torch.nn as nn
import torch
from sklearn.metrics import accuracy_score
#from c_stg.params import Params_config

import scipy.io as spio
import torch.utils.data as data_utils
from c_stg.training import *
import c_stg.models
import time
from psych.utils import acc_score

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=3, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # pt is the predicted probability
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
        
        
def choose_population(how2choose, xtr_deprocessed, xte_deprocessed, data, respath):
    th = 12;
    respath = respath + how2choose + "_"
    scaler = MinMaxScaler()
    scaler.fit(np.array(data["Age"]).reshape(-1, 1))
    th_scaled = scaler.transform(np.array([[th]]))
        
    
    if "Adhd" in how2choose: 
        trdata = data[~data.is_test]
        tedata = data[data.is_test]  
        trdata = trdata.reset_index(drop=True)
        tedata = tedata.reset_index(drop=True)
        
        datainds = ~(data["ADHD_Dx"].isna()) 
        trinds = ~(trdata["ADHD_Dx"].isna()) 
        teinds = ~(tedata["ADHD_Dx"].isna()) 
        
        data = data[datainds]
        xtr_deprocessed = xtr_deprocessed[trinds]
        xte_deprocessed = xte_deprocessed[teinds]
        
        
    # Step 1: Create datainds based on "Sex" and "Age" conditions
    if "boys" in how2choose:        
        datainds = data["Sex"] == 1
        trinds = xtr_deprocessed.Sex == 0
        teinds = xte_deprocessed.Sex == 0
    elif "girls" in how2choose:
        datainds = data["Sex"] == 2
        trinds = xtr_deprocessed.Sex == 1
        teinds = xte_deprocessed.Sex == 1
    else:
        datainds = data["Sex"] < 100  # This will effectively set all rows to True
        trinds = xtr_deprocessed.Sex < 100
        teinds = xte_deprocessed.Sex < 100
    
    # Apply age filter
    if "young" in how2choose:        
        datainds = (data["Age"] < th) & datainds  
        trinds = (xtr_deprocessed["Age"] < th_scaled[0][0]) & trinds 
        teinds = (xte_deprocessed["Age"] < th_scaled[0][0]) & teinds 
    elif "old" in how2choose:
        datainds = (data["Age"] >= th) & datainds  # Ensure correct use of parentheses
        trinds = (xtr_deprocessed["Age"] >= th_scaled[0][0]) & trinds 
        teinds = (xte_deprocessed["Age"] >= th_scaled[0][0]) & teinds 
             
       
    
    if "AdhdPos" in how2choose:        
        datainds = (data["ADHD_Dx"] == 1) & datainds  
        trinds = (xtr_deprocessed["ADHD_Dx"] == 1) & trinds 
        teinds = (xte_deprocessed["ADHD_Dx"] == 1) & teinds 
    elif "AdhdNeg" in how2choose:
        datainds = (data["ADHD_Dx"]  == 0) & datainds  # Ensure correct use of parentheses
        trinds = (xtr_deprocessed["ADHD_Dx"]  == 0) & trinds 
        teinds = (xte_deprocessed["ADHD_Dx"]  == 0) & teinds 
    
      
    data = data[datainds]
    xtr_deprocessed = xtr_deprocessed[trinds]
    xte_deprocessed = xte_deprocessed[teinds]

   
    
    return xtr_deprocessed, xte_deprocessed, data, respath, scaler
def run_cstg_analysis(xtr_deprocessed, xte_deprocessed, data, tosumconduct, contextual_feat, how2choose):      
    perf_names = {'auroc_test': 'AUROC', 'aupr_test': 'AP'} 
    feature_names = {
        'Age': 'Age',
        'Sex': 'Sex',
        'Financial_problems': 'Financial problems',    
        'Two_Parents_Household': 'Two-Parent Household', 
                     'Adoptionor_FosterCare': 'Adoptionor or FosterCare',
                     'Childrens_Aid_Service': 'Child Welfare Services ',
                     'Family_Relationship_Difficulties': 'Family-relationships Difficulties ',
                     'Between_Caregivers_Violence': 'Aggressive marital practices',
                     'Caregiver_To_Child_Violence': 'Aggressive parenting practices',
                     'Head_Injury': 'A history of head concussion ',
                     'Stimulant_Meds': 'Current stimulant medication use ',
                     'Full_Scale_IQ': 'Wechsler Full Scale IQ',
                     'rounded_WISC_Vocabulary':'Verbal cognitive performance',
                     'WISC_BlockDesign': 'Non- verbal cognitive performance',
                     'Social_Withdrawal': 'Social Withdrawal',
                     'Social_Conflicts': 'Social Conflicts',
                     'Academic_Difficulty': 'Academic Difficulty',
                     'School_Truancy': 'School Truancy',
                     'Inattention': 'Inattention',
                     'Hyperactivity_Impulsivity': 'Hyperactivity and Impulsivity symptoms',
                     'Irritability': 'Irritability',
                     'Defiance': 'Defiance',
                     'Aggresive_Conduct_Problems': 'Aggresive Conduct Problems',
                     'NonAggresive_Conduct_Problems': 'Non primarily aggressive conduct problems',
                     'Depression': 'Depression',
                     'Anxiety': 'Anxiety',
                     'Sleep_Prolems': 'Sleep Prolems',
                     'Somatization': 'Somatization',
                     'Parent_Reported_Suicidality': 'Parent-Reported suicidal ideation or behavior',
                     'Parent_Reported_SI': 'Parent-reported suicidal ideation',
                     'Parent_Reported_SB': 'Parent-reported self-harm or suicidal Attempts',
                     'Self_Reported_Sl': 'Self-reported suicidal ideation', 
                     'ParentSelfIdea': 'Parent or Self Ideation',
                     'ParentActSelfIdea': 'Parent Act or Self Ideation',
                     'ADHD_Dx': 'ADHD'}
                       				

    

    respath = ""    
    if tosumconduct:
        xtr_deprocessed['Conduct_Problems'] = xtr_deprocessed['Aggresive_Conduct_Problems'] + xtr_deprocessed['NonAggresive_Conduct_Problems']
        xte_deprocessed['Conduct_Problems'] = xte_deprocessed['Aggresive_Conduct_Problems'] + xte_deprocessed['NonAggresive_Conduct_Problems']
        xtr_deprocessed = xtr_deprocessed.drop(['Aggresive_Conduct_Problems', 'NonAggresive_Conduct_Problems'], axis=1)
        xte_deprocessed = xte_deprocessed.drop(['Aggresive_Conduct_Problems', 'NonAggresive_Conduct_Problems'], axis=1)
        feature_names["Conduct_Problems"] = 'Conduct_Problems'
        del feature_names['NonAggresive_Conduct_Problems']
        del feature_names['Aggresive_Conduct_Problems']
        respath = respath + "sumConduct_"
    xtr_deprocessed, xte_deprocessed, data, respath, scaler = choose_population(how2choose, xtr_deprocessed, xte_deprocessed, data, respath)    
    
    respath = respath + "_context_" + contextual_feat + "_cstg"
    context_features_names = []    
    contextual_features_tr = pd.DataFrame(index=xtr_deprocessed.index)
    contextual_features_te = pd.DataFrame(index=xte_deprocessed.index)
    contextual_features_tr[contextual_feat] = xtr_deprocessed[contextual_feat]
    contextual_features_te[contextual_feat] = xte_deprocessed[contextual_feat]
    if agenotfeature:        
        xtr_deprocessed = xtr_deprocessed.drop('Age', axis=1)
        xte_deprocessed = xte_deprocessed.drop('Age', axis=1)
        del feature_names['Age']
        context_features_names.append("Age")
        respath = respath + "noAge_"
    if sexnotfeature:       
        xtr_deprocessed = xtr_deprocessed.drop('Sex', axis=1)
        xte_deprocessed = xte_deprocessed.drop('Sex', axis=1)
        del feature_names['Sex']
        
        respath = respath + "noSex_"
        
    if wiscnotfeature:       
        xtr_deprocessed = xtr_deprocessed.drop('WISC_BlockDesign', axis=1)
        xte_deprocessed = xte_deprocessed.drop('WISC_BlockDesign', axis=1)
        
        xtr_deprocessed = xtr_deprocessed.drop('rounded_WISC_Vocabulary', axis=1)
        xte_deprocessed = xte_deprocessed.drop('rounded_WISC_Vocabulary', axis=1)
        del feature_names['rounded_WISC_Vocabulary']
        del feature_names['WISC_BlockDesign']
        
        respath = respath + "noWISC_"
    if adhdnotfeature:
        
        xtr_deprocessed = xtr_deprocessed.drop('ADHD_Dx', axis=1)
        xte_deprocessed = xte_deprocessed.drop('ADHD_Dx', axis=1)
        del feature_names['ADHD_Dx']
        
        respath = respath + "noADHD_"
    features = xtr_deprocessed.columns    
    #feature_names_order = np.sort([feature_names[f] for f in features])
    
    
    respath0=respath
    for ycol in ycols:
        respath = respath0 + ycol
        if not os.path.exists(respath + '/trained_model'):        
            os.mkdir(respath)
            os.mkdir(respath + '/trained_model')        
        
            
        
        xtrain = xtr_deprocessed
        ytrain = data[~data.is_test][ycol]
        xtest = xte_deprocessed
        ytest = data[data.is_test][ycol]
        feature_names_order = xtest.keys()
        # rm nans
        ytrain = ytrain.reset_index(drop=True)  # Drop the old index
        ytest = ytest.reset_index(drop=True)  # Drop the old index
        
        xtrain = xtrain.reset_index(drop=True)  # Drop the old index
        xtest = xtest.reset_index(drop=True)  # Drop the old index
        contextual_features_tr = contextual_features_tr.reset_index(drop=True)
        contextual_features_te = contextual_features_te.reset_index(drop=True)
            
        
        
        
        
        nan_indices = ytrain[ytrain.isna()].index
        ytrain = ytrain.drop(nan_indices)
        xtrain = xtrain.drop(nan_indices)
        contextual_features_tr = contextual_features_tr.drop(nan_indices)
        
        nan_indices = ytest[ytest.isna()].index
        ytest = ytest.drop(nan_indices)
        xtest = xtest.drop(nan_indices)
        contextual_features_te = contextual_features_te.drop(nan_indices)
        
        xdev = xtrain.sample(frac=0.2, random_state=42)  
        xtrain1 = xtrain.loc[~xtrain.index.isin(xdev.index)]
        
        ydev = ytrain.loc[xdev.index]
        ytrain = ytrain.loc[~xtrain.index.isin(xdev.index)]
        
        rdev = contextual_features_tr.loc[xdev.index]
        contextual_features_tr = contextual_features_tr.loc[~xtrain.index.isin(xdev.index)]
        
        xtrain = xtrain1
        pos_weight = torch.tensor([(len(ytrain)-sum(ytrain))/sum(ytrain)])
        
        train_set = data_utils.TensorDataset(torch.tensor(xtrain.to_numpy()), torch.tensor(ytrain.to_numpy()), torch.tensor(contextual_features_tr.to_numpy()))
        
        dev_set = data_utils.TensorDataset(torch.tensor(xdev.to_numpy()), torch.tensor(ydev.to_numpy()), torch.tensor(rdev.to_numpy()))
        test_set = data_utils.TensorDataset(torch.tensor(xtest.to_numpy()), torch.tensor(ytest.to_numpy()), torch.tensor(contextual_features_te.to_numpy()))
        batch_size = 32
        # Dataloaders
        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)#True)
        test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
        dev_dataloader = torch.utils.data.DataLoader(dev_set, batch_size=batch_size, shuffle=True)#True)


    
        hyper_hidden_dims = [ [50], [16, 4, 2],[10],[20]]  # units for the gates
        hidden_dims = [[10],[2], [16, 4, 2], [50]]
        learning_rates = [0.05, 0.1]
        lambdas = [0.5, 0.005, 0.05, 1] 
        for lr in learning_rates:
            for lam in lambdas:
                for hd in hidden_dims:
                    for hhd in hyper_hidden_dims:
                        
                        filename = respath + '/trained_model/wcstg_' + ycol + '_' + contextual_feat +'_' + str(lam) +'_hd' + str(hd) +'_hhd' + str(hhd) +'_lr' + str(lr) + '.mat'
                        file_path = Path(filename)
                        if file_path.is_file():
                            print("File exists")
                            continue;
                       
                        uneffective_flag = True
                        num_iter = 0
                       
                        acc = np.zeros((3, 1))
                        auc = np.zeros((3, 1))
                        ap = np.zeros((3, 1))
                        while uneffective_flag:
                           
                            # Data
                            
                            # Load model architecture
                            model = c_stg.models.__dict__["fc_stg_layered_param_modular_model_sigmoid_extension"] \
                                (xtrain.shape[1], hd, 1, 1, hyper_hidden_dim = hhd,
                                 dropout=0, sigma=0.5, include_B_in_input=False,
                                 non_param_stg=False, train_sigma=False,
                                 classification=True)
                            
                            #criterion = nn.BCELoss()
                            #criterion = FocalLoss()
                            
                            
                            # Use BCEWithLogitsLoss with pos_weight
                            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

                            optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
                            params = Cstg_params() 
                            
                            params.device = 'cpu'
                            params.output_dim = 1
                            params.num_epoch = 100
                            train_acc_array, train_loss_array, dev_acc_array, dev_loss_array, uneffective_flag, ap_train, ap_dev, acc_te_array, ap_te_array =c_stg.training.train(params, model, train_dataloader, dev_dataloader, criterion, optimizer, lam, acc_score, test_dataloader)         
                            
                            num_iter += 1
                            if uneffective_flag: print(f"Uneffective attempt#{num_iter}")
                            if num_iter == 10: uneffective_flag = False
                        
                        
                        #checkpoint = torch.load("model.pth")
                        #model.load_state_dict(checkpoint['model_state_dict'])
                        
                        model.eval()
                        acc[0], _, _, _, auc[0], ap[0] = test_process(params, model, train_dataloader,
                                                                        criterion, lam,
                                                                        acc_score)
                        acc[1], _, _, _, auc[1], ap[1] = test_process(params, model, dev_dataloader,
                                                                        criterion, lam,
                                                                        acc_score)
                        acc[2], loss_test_all, all_targets_all, labels_pred_all, auc[2], ap[2] = test_process(params, model, test_dataloader,
                                                                        criterion, lam,
                                                                        acc_score)
                        
                      
                        
                        
                        unique_r = np.unique(contextual_features_tr)  # returns a sorted array
                        #alpha_vals = np.zeros((Container.xtr.shape[1], len(unique_r)))
                        mu_vals = np.zeros((xtrain.shape[1], len(unique_r)))
                        if params.ML_model_name == "fc_stg_layered_param_modular_model_sigmoid_extension":
                            w_vals = np.zeros((xtrain.shape[1], len(unique_r)))
                        else:
                            w_vals = []
                        acc_vals_per_r = np.zeros(len(unique_r))
                        auc_vals_per_r = np.zeros(len(unique_r))
                        ap_vals_per_r = np.zeros(len(unique_r))
                        targets_per_r = [None] * len(unique_r)
                        pred_labels_per_r = [None] * len(unique_r)
                        
                        
                        ri = 0
                        for rval in np.unique(contextual_features_tr):
                            #alpha_vals[:, ri] = get_prob_alpha(params, model, np.array(rval).reshape(-1, 1))
                            if params.ML_model_name == "fc_stg_layered_param_modular_model_sigmoid_extension":
                                mu_vals[:, ri], w_vals[:, ri] =\
                                    get_prob_alpha(params, model, np.array(rval).reshape(-1, 1))
                            elif params.ML_model_name == "fc_stg_layered_param_modular_model_sigmoid":
                                mu_vals[:, ri] = get_prob_alpha(params, model, np.array(rval).reshape(-1, 1))
                            inds = [i for i, x in enumerate(contextual_features_tr == rval) if x]
                            x_test_tmp = xtest.to_numpy()[inds, :]
                            r_test_tmp = contextual_features_tr.to_numpy()[inds].reshape(-1, 1)
                            if params.classification_flag:
                                y_test_tmp = ytrain.to_numpy()[inds].reshape(-1, 1)
                            else:
                                y_test_tmp = ytrain.to_numpy()[inds]
                            # y_test_tmp = torch.empty_like(torch.tensor(r_test_tmp))
                            test_set_tmp = data_utils.TensorDataset(torch.tensor(x_test_tmp), torch.tensor(y_test_tmp),
                                                                    torch.tensor(r_test_tmp))
                            test_dataloader_tmp = torch.utils.data.DataLoader(test_set_tmp,
                                                                              batch_size=params.batch_size,
                                                                              shuffle=False)
                            acc_test, _, all_targets, labels_pred, auc_test, ap_test = test_process(params, model, test_dataloader_tmp, criterion, lam, acc_score)
                            
                            targets_per_r[ri] = all_targets
                            pred_labels_per_r[ri] = labels_pred
                            acc_vals_per_r[ri] = acc_test
                            auc_vals_per_r[ri] = auc_test
                            ap_vals_per_r[ri] = ap_test
                            ri += 1
                        
                        r_unique_unscaled = scaler.inverse_transform(unique_r.reshape(-1, 1)).reshape(-1) 
                        spio.savemat(filename,
                                     {'acc': acc, 'auc': auc, 'ap': ap, 'featurenames': feature_names_order,
                                      'ap_test_per_r': ap_vals_per_r,'acc_test_per_r': acc_vals_per_r,                      
                                      'train_acc_array': train_acc_array, 'dev_acc_array': dev_acc_array,
                                      'train_loss_array': train_loss_array, 'dev_loss_array': dev_loss_array,
                                      'unique_r': r_unique_unscaled, 'mu_vals': mu_vals,
                                      'w_vals': w_vals, 'auc_vals_per_r': auc_vals_per_r, 
                                      'context_cols': contextual_feat,  'loss_test': loss_test_all})

def run_analysis(xtr_deprocessed, xte_deprocessed, data, tosumconduct, agenotfeature, sexnotfeature,  adhdnotfeature, wiscnotfeature, how2choose):    
                
                
    perf_names = {'auroc_test': 'AUROC', 'aupr_test': 'AP'} 
    feature_names = {
        'Age': 'Age',
        'Sex': 'Sex',
        'Financial_problems': 'Financial problems',    
        'Two_Parents_Household': 'Two-Parent Household', 
                     'Adoptionor_FosterCare': 'Adoptionor or FosterCare',
                     'Childrens_Aid_Service': 'Child Welfare Services ',
                     'Family_Relationship_Difficulties': 'Family-relationships Difficulties ',
                     'Between_Caregivers_Violence': 'Aggressive marital practices',
                     'Caregiver_To_Child_Violence': 'Aggressive parenting practices',
                     'Head_Injury': 'A history of head concussion ',
                     'Stimulant_Meds': 'Current stimulant medication use ',
                     'Full_Scale_IQ': 'Wechsler Full Scale IQ',
                     'rounded_WISC_Vocabulary':'Verbal cognitive performance',
                     'WISC_BlockDesign': 'Non- verbal cognitive performance',
                     'Social_Withdrawal': 'Social Withdrawal',
                     'Social_Conflicts': 'Social Conflicts',
                     'Academic_Difficulty': 'Academic Difficulty',
                     'School_Truancy': 'School Truancy',
                     'Inattention': 'Inattention',
                     'Hyperactivity_Impulsivity': 'Hyperactivity and Impulsivity symptoms',
                     'Irritability': 'Irritability',
                     'Defiance': 'Defiance',
                     'Aggresive_Conduct_Problems': 'Aggresive Conduct Problems',
                     'NonAggresive_Conduct_Problems': 'Non primarily aggressive conduct problems',
                     'Depression': 'Depression',
                     'Anxiety': 'Anxiety',
                     'Sleep_Prolems': 'Sleep Prolems',
                     'Somatization': 'Somatization',
                     'Parent_Reported_Suicidality': 'Parent-Reported suicidal ideation or behavior',
                     'Parent_Reported_SI': 'Parent-reported suicidal ideation',
                     'Parent_Reported_SB': 'Parent-reported self-harm or suicidal Attempts',
                     'Self_Reported_Sl': 'Self-reported suicidal ideation', 
                     'ParentSelfIdea': 'Parent or Self Ideation',
                     'ParentActSelfIdea': 'Parent Act or Self Ideation',
                     'ADHD_Dx': 'ADHD'}
                       				


    respath = ""    
    if tosumconduct:
        xtr_deprocessed['Conduct_Problems'] = xtr_deprocessed['Aggresive_Conduct_Problems'] + xtr_deprocessed['NonAggresive_Conduct_Problems']
        xte_deprocessed['Conduct_Problems'] = xte_deprocessed['Aggresive_Conduct_Problems'] + xte_deprocessed['NonAggresive_Conduct_Problems']
        xtr_deprocessed = xtr_deprocessed.drop(['Aggresive_Conduct_Problems', 'NonAggresive_Conduct_Problems'], axis=1)
        xte_deprocessed = xte_deprocessed.drop(['Aggresive_Conduct_Problems', 'NonAggresive_Conduct_Problems'], axis=1)
        feature_names["Conduct_Problems"] = 'Conduct_Problems'
        del feature_names['NonAggresive_Conduct_Problems']
        del feature_names['Aggresive_Conduct_Problems']
        respath = respath + "sumConduct_"
    
    xtr_deprocessed, xte_deprocessed, data, respath, _ = choose_population(how2choose, xtr_deprocessed, xte_deprocessed, data, respath)    
    
    context_features_names = []    
    contextual_features_tr = pd.DataFrame(index=xtr_deprocessed.index)
    contextual_features_te = pd.DataFrame(index=xte_deprocessed.index)
    if wiscnotfeature:
       
        xtr_deprocessed = xtr_deprocessed.drop('WISC_BlockDesign', axis=1)
        xte_deprocessed = xte_deprocessed.drop('WISC_BlockDesign', axis=1)
        xtr_deprocessed = xtr_deprocessed.drop('rounded_WISC_Vocabulary', axis=1)
        xte_deprocessed = xte_deprocessed.drop('rounded_WISC_Vocabulary', axis=1)
        del feature_names['rounded_WISC_Vocabulary']
        del feature_names['WISC_BlockDesign']
        
        respath = respath + "noWisc_"
        
    if agenotfeature:
        contextual_features_tr["Age"] = xtr_deprocessed['Age']
        contextual_features_te["Age"] = xte_deprocessed['Age'] 
        xtr_deprocessed = xtr_deprocessed.drop('Age', axis=1)
        xte_deprocessed = xte_deprocessed.drop('Age', axis=1)
        del feature_names['Age']
        context_features_names.append("Age")
        respath = respath + "noAge_"
    if sexnotfeature:
        contextual_features_tr["Sex"] = xtr_deprocessed['Sex']
        contextual_features_te["Sex"] = xte_deprocessed['Sex']
        xtr_deprocessed = xtr_deprocessed.drop('Sex', axis=1)
        xte_deprocessed = xte_deprocessed.drop('Sex', axis=1)
        del feature_names['Sex']
        context_features_names.append("Sex")
        respath = respath + "noSex_"
    if adhdnotfeature:
        contextual_features_tr["ADHD_Dx"] = xtr_deprocessed['ADHD_Dx']
        contextual_features_te["ADHD_Dx"] = xte_deprocessed['ADHD_Dx']
        xtr_deprocessed = xtr_deprocessed.drop('ADHD_Dx', axis=1)
        xte_deprocessed = xte_deprocessed.drop('ADHD_Dx', axis=1)
        del feature_names['ADHD_Dx']
        context_features_names.append("ADHD_Dx")   
        respath = respath + "noADHD_"
    features = xtr_deprocessed.columns    
    feature_names_order = np.sort([feature_names[f] for f in features])
    
    
    respath0=respath
    for ycol in ycols:
        respath = respath0 + ycol
        
            
                
        xtrain = xtr_deprocessed
        ytrain = data[~data.is_test][ycol]
        xtest = xte_deprocessed
        ytest = data[data.is_test][ycol]
        
        # rm nans
        ytrain = ytrain.reset_index(drop=True)  # Drop the old index
        ytest = ytest.reset_index(drop=True)  # Drop the old index
        
        xtrain = xtrain.reset_index(drop=True)  # Drop the old index
        xtest = xtest.reset_index(drop=True)  # Drop the old index

        
        nan_indices = ytrain[ytrain.isna()].index
        ytrain = ytrain.drop(nan_indices)
        xtrain = xtrain.drop(nan_indices)
        
        nan_indices = ytest[ytest.isna()].index
        ytest = ytest.drop(nan_indices)
        xtest = xtest.drop(nan_indices)
        
        if not os.path.exists(respath + '/trained_model'):        
            os.mkdir(respath)
            os.mkdir(respath + '/trained_model')        
        else:
            file_path = Path(respath + '/trained_model/2024_09_2_perf.csv')

            if file_path.is_file():
                print(how2choose + " Tr overall="+ str(ytrain.shape[0]) + " positive: " + str(sum(ytrain)))
                print(how2choose + " Te overall="+ str(ytest.shape[0]) + " positive: " + str(sum(ytest)))
                continue;
                
        if os.path.exists(respath + '/trained_model/{}_perf.csv'.format(timestamp)):
            result = pd.read_csv(respath + '/trained_model/{}_perf.csv'.format(timestamp))
            feature_imp = pd.read_csv(respath + '/trained_model/{}_featureimp.p'.format(timestamp))
        else:
            result, feature_imp = classifier_utils.run_tests(tests, xtrain, ytrain, xtest, ytest, respath, save_file=timestamp, 
                                                             retrain=True)
        
        
        feature_imp['feature_name'] = [feature_names[f] for f in feature_imp.feature]
        
        sns.set_style('whitegrid')
        for test_type in tests:
            print(test_type)
            fig, ax = plt.subplots(1, figsize=(12, 12))
            sns.set(font_scale=2)
            g = sns.pointplot(y='feature_name', x='imp', ci='sd', order=feature_names_order, 
                              data=feature_imp[feature_imp.test_type==test_type], join=False, ax=ax)
            sns.set_style('whitegrid')
            ax.grid(color='lightgrey', axis='both')
            g.set_title('Feature Importance in {}'.format(test_type))#, fontsize=20)
            g.set_xlabel("Importance", fontsize=20)
            g.set_ylabel('Features', fontsize=20)
            g.tick_params(labelsize=20)
            #g.legend_.remove()
            #ax.set_yticklabels(feature_names)
            plt.tight_layout()
            fig.savefig(respath + "/features_imp_{}.pdf".format(test_type))
        
        if os.path.exists(respath + '/trained_model/{}_perfperfeature.csv'.format(timestamp)):
            print("loading for " + respath)   
            
            result_cv = pd.read_csv(respath + '/trained_model/{}_perfperfeature.csv'.format(timestamp))
            overall_ranks = pd.read_csv(respath + '/trained_model/{}_overallranks.csv'.format(timestamp))
        else:
            print("training for " + respath)   
            result_cv, overall_ranks = classifier_utils.train_per_feature(tests, xtrain, ytrain, xtest, ytest, respath, timestamp)
        
        
    
          
    
    #Save another file to keep the mean and std directly
        temp = result_cv[['test_type', 'num_features', 'name', 'value']].groupby(['test_type', 'num_features', 'name']).agg(['mean', 'std'])
        temp.reset_index(inplace=True)
        temp.columns = ['test_type', 'num_features', 'name', 'mean', 'std']
        temp.to_csv(respath + '/trained_model/{}_perfsummary.csv'.format(timestamp))   
        
    
        sns.set_style('whitegrid')
        
        for perf in perf_names.keys():
            fig, ax = plt.subplots(1, figsize=(12, 6))
            sns.set(font_scale=2)
            g = sns.pointplot(y='value', x='num_features', ci='sd', hue='test_type', hue_order=tests, 
                              data=result_cv[result_cv.name==perf], ax=ax)
            sns.set_style('whitegrid')
            ax.grid(color='lightgrey', axis='both')
            g.set_title('{} of Classifiers'.format(perf_names[perf]))#, fontsize=20)
            g.set_xlabel("Number of Features Used", fontsize=20)
            g.set_ylabel(perf_names[perf], fontsize=20)
            g.tick_params(labelsize=20)
            ax.set_ylim(0, 1)
            ax.legend(loc='lower right')
            #g.legend_.remove()
            plt.tight_layout()
            fig.savefig(respath + "/perf_{}.pdf".format(perf))
            
        fig, ax = plt.subplots(1, figsize=(12, 12))
        overall_ranks['rank'] = overall_ranks['rank'].astype(int)
        overall_ranks['feature_name'] = [feature_names[f] for f in overall_ranks.feature]
        ranksp = overall_ranks.pivot(index = "feature_name", columns = "test", values = "rank")
        sns.set(font_scale=2)
        g = sns.heatmap(ranksp)
        g.axes.set_title('Ranks of Features (0: Top)')#, fontsize=20)
        g.set_xlabel("Tests Used", fontsize=20)
        g.set_ylabel('Features', fontsize=20)
        g.tick_params(labelsize=20)
        plt.tight_layout()
        fig.savefig(respath + "/overall_ranks.pdf")
                
        for test_type in tests:
            df = result[result.test_type == test_type]
            mean_auroc = np.mean(df[df.name == 'auroc_test']['value'])
            std_auroc = np.std(df[df.name == 'auroc_test']['value'])
            mean_aupr = np.mean(df[df.name == 'aupr_test']['value'])
            std_aupr = np.mean(df[df.name == 'aupr_test']['value'])
            print("{} & {:.2f} ($\pm$ {:.2f}) & {:.2f} ($\pm$ {:.2f})".format(test_type, mean_auroc, std_auroc, mean_aupr, std_aupr))
        # Top ranks in each classifiers
        ranks_list = pd.DataFrame(columns=tests)
        for test_type in tests:
            ranks_list[test_type] = overall_ranks[overall_ranks.test == test_type].sort_values(by='rank')['feature_name'].values
            
        print(ranks_list.to_csv())
            
        import importlib
        importlib.reload(classifier_utils)
        
        classifier_utils.get_auroc_aupr(tests, xtrain, ytrain, xtest, ytest, respath, timestamp, [12, 13, 14, 15])
            
            
            
            
args = {'missing_pc_thr': 28, 'imputation_type': 'mice', 'k': 20, 'folder': 'images'}
ycols = ['Suicidality']# 'Parent_Reported_Suicidality' ,'Parent_Reported_SB','Parent_Reported_SI','Self_Reported_Sl', 'Ideation']#'Parent_Reported_SI', 'Parent_Reported_SB', 'Self_Reported_Sl',,
        				 	 

tests = [ "xgb", "rf", "gb", "l1" ]#"ebm",
#timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
#timestamp
timestamp = '2024_09_2'

if not os.path.exists(args['folder']):
    os.makedirs(args['folder'])
    



#data = pd.read_csv('dataset_240120.csv', na_values=['#NULL!', '', ' '])
data = pd.read_csv('../data/data3.csv', na_values=['#NULL!', '', ' '])
# data 3 is with suicidal vars being 0 or 1

if os.path.exists('dataset_df_impute_{}.p'.format(timestamp)):
    df_impute = pickle.load(open('dataset_df_impute_{}.p'.format(timestamp), 'rb'))
    xtr_deprocessed = pickle.load(open('dataset_xtr_{}.p'.format(timestamp), 'rb'))
    xte_deprocessed = pickle.load(open('dataset_xte_{}.p'.format(timestamp), 'rb'))
    
    
else:
    df_impute, xtr_deprocessed, xte_deprocessed = utils.process_pipeline(data, 'features_dict_current1.csv', ycols,
                                                      missing_pc_thr=args['missing_pc_thr'],
                                                      imputation_type=args['imputation_type'],
                                                      k=args['k'], verbose=False)

    pickle.dump(df_impute, open('dataset_df_impute_{}.p'.format(timestamp), 'wb'))
    pickle.dump(xtr_deprocessed, open('dataset_xtr_{}.p'.format(timestamp), 'wb'))
    pickle.dump(xte_deprocessed, open('dataset_xte_{}.p'.format(timestamp), 'wb'))


# make an Ideation variable
nan_indices = data[(data["Parent_Reported_SI"].isna()) | (data["Self_Reported_Sl"].isna())].index
Ideation = (data["Parent_Reported_SI"].fillna(False).astype(bool) | data["Self_Reported_Sl"].fillna(False).astype(bool)).astype(float)
Ideation[nan_indices] = np.nan

# make a Suicidality variable
nan_indices = data[(data["Parent_Reported_SI"].isna()) | (data["Parent_Reported_SB"].isna()) | (data["Self_Reported_Sl"].isna())].index
Suicidality = (data["Parent_Reported_SI"].fillna(False).astype(bool) | data["Parent_Reported_SB"].fillna(False).astype(bool) | data["Self_Reported_Sl"].fillna(False).astype(bool)).astype(float)
Suicidality[nan_indices] = np.nan

data["Ideation"] = Ideation
data["Suicidality"] = Suicidality

# agenotfeature = True
# sexnotfeature = True
# adhdnotfeature = True
# tosumconduct = True
# run_cstg_analysis(xtr_deprocessed, xte_deprocessed, data, tosumconduct, "Age", "girls")
# run_cstg_analysis(xtr_deprocessed, xte_deprocessed, data, tosumconduct, "Age", "boys")

# # sum conduct, by age and sex, adhd is not a feature
# agenotfeature = True
# sexnotfeature = False
# adhdnotfeature = True
# tosumconduct = True
# run_cstg_analysis(xtr_deprocessed, xte_deprocessed, data, tosumconduct, "Age", "all")

# tosumconduct = True
# agenotfeature= True
# sexnotfeature = True
# adhdnotfeature = True
# run_analysis(xtr_deprocessed, xte_deprocessed, data, tosumconduct, agenotfeature, 
#             sexnotfeature,  adhdnotfeature, 'old_girls_AdhdPos')
# run_analysis(xtr_deprocessed, xte_deprocessed, data, tosumconduct, agenotfeature, 
#             sexnotfeature,  adhdnotfeature, 'young_boys_AdhdPos')
# run_analysis(xtr_deprocessed, xte_deprocessed, data, tosumconduct, agenotfeature, 
#             sexnotfeature,  adhdnotfeature, 'young_boys_AdhdNeg')
# run_analysis(xtr_deprocessed, xte_deprocessed, data, tosumconduct, agenotfeature, 
#             sexnotfeature,  adhdnotfeature, 'old_boys_AdhdPos')
# run_analysis(xtr_deprocessed, xte_deprocessed, data, tosumconduct, agenotfeature, 
#             sexnotfeature,  adhdnotfeature, 'old_boys_AdhdNeg')

# run_analysis(xtr_deprocessed, xte_deprocessed, data, tosumconduct, agenotfeature, 
#             sexnotfeature,  adhdnotfeature, 'young_girls_AdhdPos')
# run_analysis(xtr_deprocessed, xte_deprocessed, data, tosumconduct, agenotfeature, 
#             sexnotfeature,  adhdnotfeature, 'young_girls_AdhdNeg')

# run_analysis(xtr_deprocessed, xte_deprocessed, data, tosumconduct, agenotfeature, 
#             sexnotfeature,  adhdnotfeature, 'old_girls_AdhdNeg')

# tosumconduct = True
# agenotfeature= True
# sexnotfeature = False
# adhdnotfeature = True
# run_analysis(xtr_deprocessed, xte_deprocessed, data, tosumconduct, agenotfeature, 
#             sexnotfeature,  adhdnotfeature, 'young_AdhdPos')
# run_analysis(xtr_deprocessed, xte_deprocessed, data, tosumconduct, agenotfeature, 
#             sexnotfeature,  adhdnotfeature, 'old_AdhdPos')
# run_analysis(xtr_deprocessed, xte_deprocessed, data, tosumconduct, agenotfeature, 
#             sexnotfeature,  adhdnotfeature, 'young_AdhdNeg')
# run_analysis(xtr_deprocessed, xte_deprocessed, data, tosumconduct, agenotfeature, 
#             sexnotfeature,  adhdnotfeature, 'old_AdhdNeg')

# tosumconduct = True
# agenotfeature= False
# sexnotfeature = True
# adhdnotfeature = True
# run_analysis(xtr_deprocessed, xte_deprocessed, data, tosumconduct, agenotfeature, 
#             sexnotfeature,  adhdnotfeature, 'boys_AdhdPos')
# run_analysis(xtr_deprocessed, xte_deprocessed, data, tosumconduct, agenotfeature, 
#             sexnotfeature,  adhdnotfeature, 'girls_AdhdPos')
# run_analysis(xtr_deprocessed, xte_deprocessed, data, tosumconduct, agenotfeature, 
#             sexnotfeature,  adhdnotfeature, 'boys_AdhdNeg')
# run_analysis(xtr_deprocessed, xte_deprocessed, data, tosumconduct, agenotfeature, 
#             sexnotfeature,  adhdnotfeature, 'girls_AdhdNeg')

# tosumconduct = True
# agenotfeature= False
# sexnotfeature = False
# adhdnotfeature = True
# run_analysis(xtr_deprocessed, xte_deprocessed, data, tosumconduct, agenotfeature, 
#             sexnotfeature,  adhdnotfeature, 'all_AdhdPos')
# run_analysis(xtr_deprocessed, xte_deprocessed, data, tosumconduct, agenotfeature, 
#             sexnotfeature,  adhdnotfeature, 'all_AdhdNeg')
# sum conduct, age and sex are features, adhd is not, all population
agenotfeature = False
sexnotfeature = False
adhdnotfeature = True
tosumconduct = True
wiscnotfeature = True
# run_analysis(xtr_deprocessed, xte_deprocessed, data, tosumconduct, agenotfeature, 
#             sexnotfeature,  adhdnotfeature, wiscnotfeature, 'all')


# sum conduct, by sex, age is a feature, adhd is not, boys and girls
agenotfeature = False
sexnotfeature = True
adhdnotfeature = True
tosumconduct = True
wiscnotfeature = True
run_analysis(xtr_deprocessed, xte_deprocessed, data, tosumconduct, agenotfeature, 
            sexnotfeature,  adhdnotfeature, wiscnotfeature, 'girls')
run_analysis(xtr_deprocessed, xte_deprocessed, data, tosumconduct, agenotfeature, 
            sexnotfeature,  adhdnotfeature, wiscnotfeature, 'boys')

# sum conduct, by age, sex is a feature, adhd is not
agenotfeature = False
sexnotfeature = False
adhdnotfeature = True
tosumconduct = True
wiscnotfeature = True
run_analysis(xtr_deprocessed, xte_deprocessed, data, tosumconduct, agenotfeature, 
            sexnotfeature,  adhdnotfeature, wiscnotfeature, 'young')
run_analysis(xtr_deprocessed, xte_deprocessed, data, tosumconduct, agenotfeature, 
            sexnotfeature,  adhdnotfeature, wiscnotfeature, 'old')


# sum conduct, by age and sex, adhd is not a feature
agenotfeature = False
sexnotfeature = True
adhdnotfeature = True
tosumconduct = True
wiscnotfeature = True
run_analysis(xtr_deprocessed, xte_deprocessed, data, tosumconduct, agenotfeature, 
            sexnotfeature,  adhdnotfeature, wiscnotfeature, 'old_girls')
run_analysis(xtr_deprocessed, xte_deprocessed, data, tosumconduct, agenotfeature, 
            sexnotfeature,  adhdnotfeature, wiscnotfeature, 'young_girls')
run_analysis(xtr_deprocessed, xte_deprocessed, data, tosumconduct, agenotfeature, 
            sexnotfeature,  adhdnotfeature, wiscnotfeature, 'old_boys')
run_analysis(xtr_deprocessed, xte_deprocessed, data, tosumconduct, agenotfeature, 
            sexnotfeature,  adhdnotfeature, wiscnotfeature, 'young_boys')

