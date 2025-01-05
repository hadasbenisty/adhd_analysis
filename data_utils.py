import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def preprocess(dataframe, types_dict, return_scaler=False):
    column_names = sorted(dataframe.columns)
    dataframe = dataframe[column_names]
    data = np.asarray(dataframe)
    all_names_dict = [types_dict[k]['name'] for k in range(len(types_dict))]  # sorted(all_names_dict) is equal to sorted(column_names)
    # Construct the data matrices
    data_complete = []
    column_names_new = []
    cont_columns = []
    for i, column in enumerate(column_names):
        k = [kk for kk in range(len(all_names_dict)) if all_names_dict[kk] == column][0]  # todo: use index() instead
        # print(types_dict[k], column)
        if types_dict[k]['type'] == 'cat' or types_dict[k]['type'] == 'ord':
            # Get categories
            cat_data = [x for x in data[:, i] if not np.isnan(x)]
            categories, indexes = np.unique(cat_data, return_inverse=True)
            # Transform categories to a vector of 0:n_categories
            print(column)
            assert len(categories) == int(types_dict[k]['dim'])
            new_categories = np.arange(len(categories))
            cat_data = new_categories[indexes]
            if types_dict[k]['type'] == 'cat':
                # Create one hot encoding for the categories
                aux = np.zeros([np.shape(data)[0], len(new_categories)])
                aux[np.where(~np.isnan(data[:, i]))[0], cat_data] = 1
                aux[np.where(np.isnan(data[:, i]))[0], :] = np.nan
            else:
                # Create thermometer encoding for the categories
                aux = np.zeros([np.shape(data)[0], 1+len(new_categories)])
                aux[:, 0] = 1
                aux[np.where(np.logical_not(np.isnan(data[:, i])))[0], 1+cat_data] = -1
                aux = np.cumsum(aux, 1)
                aux[np.where(np.isnan(data[:, i]))[0], :] = np.nan
                aux = aux[:, :-1]
            data_complete.append(aux)
            column_names_new.extend([column_names[i] + '_' + str(categories[kk]) for kk in range(len(new_categories))])
        else:
            data_complete.append(np.transpose([data[:, i]]))  #/*/ transpose does nothing...
            column_names_new.append(column_names[i])
            cont_columns.append(column_names[i])

    data_new = np.concatenate(data_complete, axis=1)
    data_new = pd.DataFrame(data_new, columns=column_names_new)

    # Scale continuous data between 0 and 1
    scaler = MinMaxScaler()
    if len(cont_columns) > 0:
        scaler.fit(data_new[cont_columns])
        data_new[cont_columns] = scaler.transform(data_new[cont_columns])
    if return_scaler:
        return data_new, scaler
    return data_new


###/*/ has bug: assumes column_names_old is sorted!!!
def deprocess(dataframe, types_dict, column_names_old, verbose=True):
    temp = column_names_old[:]  # make shallow copy
    column_names_old.sort()
    #column_names = sorted(dataframe.columns) # This shouldn't be here! causes problems with 'c100' and 'c1' categorical value
    #dataframe = dataframe[column_names]
    data = np.asarray(dataframe)
    ind_ini = 0
    output = []
    for i, column in enumerate(column_names_old):
        types_dict_curr = [types_dict[kk] for kk in range(len(types_dict)) if types_dict[kk]['name'] == column][0]
        if verbose:
            print('types_dict_curr', types_dict_curr)
        ind_end = ind_ini + int(types_dict_curr['dim'])
        if verbose:
            print('cols:', ind_ini, ind_end, 'columns:', dataframe.columns[ind_ini:ind_end])
        if types_dict_curr['type'] == 'cat':
            output.append(np.reshape(np.argmax(data[:, ind_ini:ind_end], 1), [-1, 1]))
        elif types_dict_curr['type'] == 'ord':
            output.append(np.reshape(np.sum(data[:, ind_ini:ind_end], 1) - 1, [-1, 1]))
        else:
            output.append(data[:, ind_ini:ind_end])
        ind_ini = ind_end

    data_mat = np.concatenate(output, 1)
    data_deprocessed = pd.DataFrame(data_mat, columns=column_names_old)
    data_deprocessed = data_deprocessed[temp]
    return data_deprocessed


def generate_feature_dict(data_csv, feature_csv, output_csv):
    '''''Given a csv file containing data and a csv list of feature names, output another csv with feature information,
    such as: name, type, dim, nclass'''''
    df = pd.read_csv(data_csv, na_values=[' ', '', '#NULL!'])
    features = pd.read_csv(feature_csv)
    output = pd.DataFrame(columns=['name', 'type', 'dim', 'nclass'])
    for f in features.name:
        try:
            if f in df.columns and \
                    df[f].dtype in [np.dtype('float64'), np.dtype('float32'), np.dtype('int64'), np.dtype('int32')]:
                u = df[f].unique()
                if f == 'Sex' or f.startswith('CurrMedTyp') or f.startswith('PrevMedTyp'):
                    ndim = (~np.isnan(u)).sum()
                    output = output.append({'name': f, 'type': 'cat', 'dim': ndim}, ignore_index=True, sort=False)
                elif f == 'AgeContinuous' or f.startswith('WISC') or f.startswith('CurrMedDur') or f.startswith('CurrMedDos'):
                    output = output.append({'name': f, 'type': 'pos', 'dim': 1}, ignore_index=True, sort=False)
                elif np.array_equal([0, 1], np.sort(u[~np.isnan(u)])):
                    output = output.append({'name': f, 'type': 'bin', 'dim': 1}, ignore_index=True, sort=False)
                elif df[f].dtype in [np.dtype('float64'), np.dtype('float32')]:
                    if np.all([np.isnan(i) or i.is_integer() for i in df[f].unique()]):
                        ndim = (~np.isnan(u)).sum()
                        output = output.append({'name': f, 'type': 'ord', 'dim': ndim}, ignore_index=True, sort=False)
                    else:
                        output = output.append({'name': f, 'type': 'pos', 'dim': 1}, ignore_index=True, sort=False)
                else:
                    ndim = (~np.isnan(u)).sum()
                    output = output.append({'name': f, 'type': 'ord', 'dim': ndim}, ignore_index=True, sort=False)
        except TypeError as e:
            print(e)
            print(f)
            print(df[f].unique())

    output['nclass'] = output['dim']
    output.to_csv(output_csv)
    return output
