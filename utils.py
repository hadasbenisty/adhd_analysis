import pandas as pd
import numpy as np
import fancyimpute
import pydot
from scipy.stats import entropy
import sklearn
import ast
import csv
import data_utils as du


EDGESTRS = ['', '-->', '<--', '<->', 'o->', '<-o', 'o-o', '---']  # empty string = no edge
EDGENUM_TO_STR = dict(zip(range(len(EDGESTRS)), EDGESTRS))
EDGESTR_TO_NUM = dict(zip(EDGESTRS, range(len(EDGESTRS))))
REVERSE_EDGESTR = {
    ''   : '',
    '-->': '<--',
    '<--': '-->',
    '<->': '<->',
    'o->': '<-o',
    '<-o': 'o->',
    'o-o': 'o-o',
    '---': '---'
}
REVERSE_EDGENUM = {EDGESTR_TO_NUM[k]: EDGESTR_TO_NUM[v] for (k, v) in REVERSE_EDGESTR.items()}

EDGESTR_INCLUDED_STRS = {
    ''   : [''],
    '-->': ['-->'],
    '<->': ['<->'],
    'o->': ['-->', '<->', 'o->'],
    'o-o': ['-->', '<->', 'o->', '<--', '<-o', 'o-o'],
    '<--': ['<--'],
    '<-o': ['<--', '<->', '<-o'],
    '---': ['-->', '<--']
}
EDGENUM_INCLUDED_NUMS = {EDGESTR_TO_NUM[k]: [EDGESTR_TO_NUM[s] for s in lis] for (k, lis) in EDGESTR_INCLUDED_STRS.items()}


def edge_is_subset_of_edge(edge1, edge2):
    if type(edge1) != type(edge2):
        raise Exception("Edges should both be either str or int")

    if type(edge1) == str:
        return edge1 in EDGESTR_INCLUDED_STRS[edge2]
    else:
        return int(edge1) in EDGENUM_INCLUDED_NUMS[int(edge2)]


# fancyimpute does weighted knn!
def impute_knn(dataframe, k, return_imputer=False):
    imputer = fancyimpute.KNN(k=k)
    df_imputed = imputer.fit_transform(dataframe)
    if return_imputer:
        return pd.DataFrame(data=df_imputed, columns=dataframe.columns), imputer
    return pd.DataFrame(data=df_imputed, columns=dataframe.columns)


# how to do MICE? use IterativeImputer, or try installing the conda (brittain) version
# run IterativeImputer several times (5x), run GFCI on each,
# and aggregate the causal graphs (take the mode for each potential edge)
# see bottom code snippet from https://pypi.org/project/fancyimpute/
def impute_mice(dataframe, n_imputations, return_imputer=False, estimator=sklearn.linear_model.BayesianRidge()):
    imputed_list = []
    imputers = []
    for i in range(n_imputations):
        print("Starting imputation number {}".format(i))
        imputer = fancyimpute.IterativeImputer(sample_posterior=True, random_state=i)
        imputed_array = imputer.fit_transform(dataframe)
        imputed_list.append(pd.DataFrame(data=imputed_array, columns=dataframe.columns, index=dataframe.index))
        imputers.append(imputer)
    if return_imputer:
        return imputed_list, imputers
    return imputed_list


def save_graph_from_pydot_str(s, filename):
    if filename[-4:] != ".png":
        filename += ".png"
    graphs = pydot.graph_from_dot_data(s)
    graphs[0].write_png(filename)


def edges_from_pydot_str(s):
    graphs = pydot.graph_from_dot_data(s)
    edges = graphs[0].get_edges()
    return [[x.get_source().translate({ord('"'): ''}), x.get_destination().translate({ord('"'): ''})] for x in edges]


# input: list of edges
# TODO: refactor code: create enum for edge types, and create a helper function edgeset_to_adjmat
# possible settings: most general, most common, 0/preserved, 1/highest, 2/majority
def aggregate_edgesets(edgeset_list, setting="most general", entropy_threshold=0.5):
    if setting in ["most general", "most common"]:
        edge_histogram = parse_edges(edgeset_list, "discrete")
    else:  # look at the probabilities of each edge
        try:
            edge_histogram = parse_edges(edgeset_list, "prob")
        except Exception:
            raise Exception("This setting is invalid for this input format of edgeset_list. "
                            "Put setting='most common' or 'most general' instead.")
    output = {'edgeset': [], 'pydot': '', 'edgeset_wo_prob': [], 'pydot_wo_prob': ''}
    for pair in edge_histogram.keys():
        if setting == "most common":
            # choose the most commonly occurring edge type out of the 6 types + no edge.

            no_edge_freq = len(edgeset_list) - sum(edge_histogram[pair].values())
            edgestr, freq = max(edge_histogram[pair].items(), key=lambda x: x[1])
            if freq < no_edge_freq:
                edgestr = None
        elif setting == "most general":
            # the most conservative setting.
            # if no graph has an edge between a given pair, then choose no edge.
            # if any graph has an edge between a given pair, choose the most general one as follows:
            # o-o (most general) is the union of --> and <-> and <--
            # o-> is the union of --> and <->
            # <-o is the union of <-- and <->
            # note that --> and <-- are disjoint.
            edgestr = find_most_general_edge(edge_histogram[pair])
        elif setting in ["preserved", 0]:
            # an edge is kept and its orientation is chosen based on the highest probability.
            no_edge_prob = edge_histogram[pair]['']
            edge_histogram[pair][''] = 0  # remove the possibility of no edge being chosen
            edgestr = max(edge_histogram[pair], key=edge_histogram[pair].get)  # Get key of dictionary w/ highest prob
            ties = check_ties_among_edges(edge_histogram[pair], edgestr)
            if len(ties) > 0:
                if edgestr != "":
                    ties.append(edgestr)
                edgestr = find_most_general_edge({k: v for k, v in edge_histogram[pair].items() if k in ties})
            edge_histogram[pair][''] = no_edge_prob  # put back the original probability

        elif setting in ["highest", 1]:
            # an edge is kept the same way the preserved ensemble one does except when [no edge]'s probability is the
            # highest one, then the edge is ignored

            # Add the rest of the noedge probability
            edge_histogram[pair][''] += len(edgeset_list) - round(sum(edge_histogram[pair].values()))

            edgestr = max(edge_histogram[pair], key=edge_histogram[pair].get)  # Get key of dictionary w/ highest prob
            ties = check_ties_among_edges(edge_histogram[pair], edgestr)
            if len(ties) > 0:
                if edgestr != "":
                    ties.append(edgestr)
                edgestr = find_most_general_edge({k: v for k, v in edge_histogram[pair].items() if k in ties})

            if edgestr == "":
                edgestr = None
        elif setting in ["majority", 2]:
            # the edge is kept only if its chosen orientations' probability is more than 0.5

            # Add the rest of the noedge probability
            edge_histogram[pair][''] += len(edgeset_list) - round(sum(edge_histogram[pair].values()))

            edgestr = max(edge_histogram[pair], key=edge_histogram[pair].get)  # Get key of dictionary w/ highest prob
            ties = check_ties_among_edges(edge_histogram[pair], edgestr)
            if len(ties) > 0:
                if edgestr != "":
                    ties.append(edgestr)
                edgestr = find_most_general_edge({k: v for k, v in edge_histogram[pair].items() if k in ties})
                edge_histogram[pair][edgestr] = sum([edge_histogram[pair][tie_col] for tie_col in ties])

            if edgestr == "" or edge_histogram[pair][edgestr] <= 0.5 * len(edgeset_list):
                edgestr = None
        elif setting in ["entropy_threshold", 3]:
            # similar to preserved aggregation except that it's considered no edge if the entropy > threshold
            if entropy(list(edge_histogram[pair].values())) <= entropy_threshold:
                no_edge_prob = edge_histogram[pair]['']
                edge_histogram[pair][''] = 0  # remove the possibility of no edge being chosen
                edgestr = max(edge_histogram[pair], key=edge_histogram[pair].get)  # Get key of dictionary w/ highest prob
                ties = check_ties_among_edges(edge_histogram[pair], edgestr)
                if len(ties) > 0:
                    if edgestr != "":
                        ties.append(edgestr)
                    edgestr = find_most_general_edge({k: v for k, v in edge_histogram[pair].items() if k in ties})
                edge_histogram[pair][''] = no_edge_prob  # put back the original probability
            else:
                edgestr = None
        elif setting in ["tree", 4]:
            # Stage 1: compare accumulated probability of having edge vs no edge
            if edge_histogram[pair][''] >= 0.5 * len(edgeset_list):
                edgestr = None

            # Stage 2: check whether the edge is more likely to be o-o, <->, <-o + <--, o-> + -->
            else:
                stage2_cats = {'o-o': ['o-o'], '<->': ['<->'], 'left': ['<--', '<-o'], 'right': ['-->', 'o->']}
                stage2_probs = {
                    'o-o': edge_histogram[pair]['o-o'],
                    '<->': edge_histogram[pair]['<->'],
                    'left': edge_histogram[pair]['<--'] + edge_histogram[pair]['<-o'],
                    'right': edge_histogram[pair]['-->'] + edge_histogram[pair]['o->']
                }
                stage2_maxcat = [max(stage2_probs, key=stage2_probs.get)]
                stage2_maxcat += check_ties_among_edges(stage2_probs, stage2_maxcat[0])

                # For next comparison, we should compare among these:
                sub_edgetypes = []
                for cat in stage2_maxcat:
                    sub_edgetypes += stage2_cats[cat]

                sub_hist = {k: edge_histogram[pair][k] for k in sub_edgetypes}

                # Stage 3: find max in the sub-direction
                edgestr = max(sub_hist, key=sub_hist.get)
                ties = check_ties_among_edges(sub_hist, edgestr)
                if len(ties) > 0:
                    if edgestr != "":
                        ties.append(edgestr)
                    edgestr = find_most_general_edge({k: v for k, v in sub_hist.items() if k in ties})
        else:
            raise Exception('Setting is not available.')

        if edgestr is not None:
            edgestr_wo_prob = pair[0] + ' ' + edgestr + ' ' + pair[1] + ' '
            output['edgeset_wo_prob'].append(edgestr_wo_prob)

            # include probabilities to edge information
            edgestr_details = edgestr_wo_prob
            pydot_details = '{} - {}'.format(pair[0], pair[1])
            sum_prob = sum(edge_histogram[pair].values())
            for edge_type in edge_histogram[pair]:
                if edge_histogram[pair][edge_type] > 0:
                    pct = edge_histogram[pair][edge_type]/sum_prob
                    if edge_type == "":
                        edgestr_details += '[{}]:{};'.format('no edge', pct)
                        pydot_details += '\n[{}]:{}'.format('no edge', pct)
                    else:
                        edgestr_details += '[{} {} {}]:{};'.format(pair[0], edge_type, pair[1], pct)  # normalize
                        pydot_details += '\n[{}]:{}'.format(edge_type, pct)
            output['edgeset'].append(edgestr_details)

            arrowtail = 'odot' if edgestr[0] == 'o' \
                        else 'normal' if edgestr[0] == '<' \
                        else 'none'
            arrowhead = 'odot' if edgestr[-1] == 'o' \
                        else 'normal' if edgestr[-1] == '>' \
                        else 'none'

            output['pydot'] += ' "{}" -> "{}" [dir=both, arrowtail={}, arrowhead={}, label="{}"]; \n'.\
                format(pair[0], pair[1], arrowtail, arrowhead, pydot_details)
            output['pydot_wo_prob'] += ' "{}" -> "{}" [dir=both, arrowtail={}, arrowhead={}]; \n'. \
                format(pair[0], pair[1], arrowtail, arrowhead)

    output['pydot'] = 'digraph g {\n' + output['pydot'] + '}'
    output['pydot_wo_prob'] = 'digraph g {\n' + output['pydot_wo_prob'] + '}'

    return output


def parse_edges(edgeset_list, count):
    edge_histogram = {}

    if count == "prob":  # add up the probability
        for edgeset in edgeset_list:
            for edge in edgeset:
                if len(edge) > 1:
                    mparent, _, mchild = edge.split(' ')[:3]  # Get the main parent, edgestr, child in case for no edge
                    idx_start = edge.find('[')
                    if idx_start < 0:
                        raise Exception("This setting is invalid for this input format of edgeset_list. "
                                        "Put count='discrete' instead.")
                    for subedge in edge[idx_start:].split(";"):
                        if subedge == "":
                            continue
                        edge_type, prob = subedge.split(":")
                        prob = float(prob)
                        if edge_type == "[no edge]":
                            edgestr = ""
                            parent = mparent
                            child = mchild
                        else:
                            parent, edgestr, child = edge_type[1:-1].split(' ')  # Subset edge_type to remove the []

                        if (parent, child) not in edge_histogram and (child, parent) not in edge_histogram:
                            edge_histogram[(parent, child)] = dict(zip(EDGESTRS, [0] * len(EDGESTRS)))

                        if (parent, child) in edge_histogram:
                            edge_histogram[(parent, child)][edgestr] += prob
                        else:  # (child, parent) is in edge_histogram keys, so turn it into a backwards edge
                            edgestr = REVERSE_EDGESTR[edgestr]
                            edge_histogram[(child, parent)][edgestr] += prob
    else:  # count existence of edges
        for edgeset in edgeset_list:
            for edge in edgeset:
                if len(edge) > 1:
                    parent, edgestr, child = edge.split(' ')[:3]
                    if (parent, child) not in edge_histogram and (child, parent) not in edge_histogram:
                        edge_histogram[(parent, child)] = dict(zip(EDGESTRS, [0] * len(EDGESTRS)))

                    if (parent, child) in edge_histogram:
                        edge_histogram[(parent, child)][edgestr] += 1
                    else:  # (child, parent) is in edge_histogram keys, so turn it into a backwards edge
                        edgestr = REVERSE_EDGESTR[edgestr]
                        edge_histogram[(child, parent)][edgestr] += 1

    # Check that for each key in histogram, the sum of the count/prob should be equal to each other
    for pair in edge_histogram:
        sum_prob = sum(edge_histogram[pair].values())
        if round(sum_prob, 2) > len(edgeset_list):
            print(edge_histogram[pair])
            raise Exception("This shouldn't be possible. The sum of probabilities is more than the numbers of graph.")
        elif sum_prob < len(edgeset_list):
            # Sum is not equal to number of graphs
            edge_histogram[pair][""] += len(edgeset_list) - sum_prob

    return edge_histogram


def find_most_general_edge(edges):
    # if any graph has an edge between a given pair, choose the most general one as follows:
    # o-o (most general) is the union of --> and <-> and <--
    # o-> is the union of --> and <->
    # <-o is the union of <-- and <->
    # note that --> and <-- are disjoint.

    # define possibilities
    possible = {'-->': False, '<->': False, '<--': False, '---': False}
    if 'o-o' in edges and edges['o-o'] > 0:
        possible['-->'] = True
        possible['<->'] = True
        possible['<--'] = True
    if 'o->' in edges and edges['o->'] > 0:
        possible['-->'] = True
        possible['<->'] = True
    if '<-o' in edges and edges['<-o'] > 0:
        possible['<--'] = True
        possible['<->'] = True
    if '-->' in edges and edges['-->'] > 0:
        possible['-->'] = True
    if '<->' in edges and edges['<->'] > 0:
        possible['<->'] = True
    if '<--' in edges and edges['<--'] > 0:
        possible['<--'] = True
    if '---' in edges and edges['---'] > 0:
        possible['---'] = True

    if possible['-->'] and possible['<--']:
        edgestr = 'o-o'
    elif possible['-->'] and possible['<->']:
        edgestr = 'o->'
    elif possible['<--'] and possible['<->']:
        edgestr = '<-o'
    elif possible['---']:
        edgestr = '---'
    elif possible['-->']:
        edgestr = '-->'
    elif possible['<->']:
        edgestr = '<->'
    elif possible['<--']:
        edgestr = '<--'
    else:
        raise Exception("This should never happen because all cases should be covered")

    return edgestr


def check_ties_among_edges(edges, curmax):
    ties = []
    for other_edge in edges:
        if other_edge != "" and other_edge != curmax and \
                edges[other_edge] == edges[curmax]:  # same probability
            ties.append(other_edge)
    return ties


def edgeset_to_adjmat(edgeset: object, setting: object = "int") -> object:
    vertices = set()
    for edge in edgeset:
        if len(edge) > 0:
            v1, _, v2 = edge.split(' ')[:3]  # IGNORES LESS LIKELY ONES!!
            vertices.add(v1)
            vertices.add(v2)
    vertices = list(vertices)
    vertices.sort()
    N = len(vertices)
    if setting == "int":
        adjmat = pd.DataFrame(data=np.zeros((N, N)), columns=vertices, index=vertices, dtype=int)
    elif setting == "str":
        adjmat = pd.DataFrame(data=np.empty((N, N), dtype=str), columns=vertices, index=vertices, dtype=str)
    else:
        raise Exception("setting should be int or str")

    for edge in edgeset:
        if len(edge.split(' ')) < 3:
            continue
        v1, edgestr, v2 = edge.split(' ')[:3]
        if setting == "int":
            if adjmat[v1][v2] != 0 or adjmat[v2][v1] != 0:
                raise Exception(
                    "This should never happen. Edgeset has multiple edges between the same pair of vertices.")
        elif setting == "str":
            if adjmat[v1][v2] != '' or adjmat[v2][v1] != '':
                raise Exception(
                    "This should never happen. Edgeset has multiple edges between the same pair of vertices.")

        if setting == "int":
            adjmat[v1][v2] = EDGESTR_TO_NUM[edgestr]
            adjmat[v2][v1] = EDGESTR_TO_NUM[REVERSE_EDGESTR[edgestr]]
        elif setting == "str":
            adjmat[v1][v2] = edgestr
            adjmat[v2][v1] = REVERSE_EDGESTR[edgestr]

    return adjmat


# edgeset format: the format returned from tetrad.getEdges(), which is a list of strings like "X1 o-> X2".
# This one only counts edges as matching if the edges are exactly the same. (i.e. "strict")
# It does not take into account if one is a subset of the other (e.g. --> is a subset of o->).
def SHD_strict_edge_match(edgeset1, edgeset2):
    adjmat1 = edgeset_to_adjmat(edgeset1)
    adjmat2 = edgeset_to_adjmat(edgeset2)

    if list(adjmat1.columns) != list(adjmat2.columns):
        # raise Exception("Debug statement: vertices are different.")
        # Add each other's missing columns for comparison
        for col in np.setdiff1d(adjmat1.columns, adjmat2.columns):
            adjmat2[col] = 0
        for col in np.setdiff1d(adjmat2.columns, adjmat1.columns):
            adjmat1[col] = 0

        # Reorder the columns
        all_cols = np.union1d(adjmat1.columns, adjmat2.columns)
        adjmat1 = adjmat1[all_cols]
        adjmat2 = adjmat2[all_cols]

        # Add each other's missing rows for comparison
        for col in np.setdiff1d(list(adjmat1.index), list(adjmat2.index)):
            adjmat2.loc[col] = 0
        for col in np.setdiff1d(list(adjmat2.index), list(adjmat1.index)):
            adjmat1.loc[col] = 0

        # Reorder the rows
        all_rows = np.union1d(list(adjmat1.index), list(adjmat2.index))
        adjmat1 = adjmat1.reindex(all_rows)
        adjmat2 = adjmat2.reindex(all_rows)

    dist = 0
    for i in range(len(adjmat1.columns)):
        for j in range(i):
            v1 = adjmat1.columns[i]
            v2 = adjmat1.columns[j]

            if adjmat1[v1][v2] != adjmat2[v1][v2]:
                dist += 1
    return dist


# This one is less strict than the above. It counts edges as matching if the DAG edge is a subset of the PAG edge.
def SHD_DAG_and_PAG(edgeset_DAG, edgeset_PAG):
    adjmat_DAG = edgeset_to_adjmat(edgeset_DAG)
    adjmat_PAG = edgeset_to_adjmat(edgeset_PAG)

    dist = 0
    for i in range(len(adjmat_DAG.columns)):
        for j in range(i):
            v1 = adjmat_DAG.columns[i]
            v2 = adjmat_DAG.columns[j]
            if v1 not in list(adjmat_PAG.columns) or v2 not in list(adjmat_PAG.columns):
                if adjmat_DAG[v1][v2] != 0:
                    # if PAG doesn't contain one of the vertices, then it only matches if DAG has no edge between v1 and v2.
                    dist += 1
            elif not edge_is_subset_of_edge(adjmat_DAG[v1][v2], adjmat_PAG[v1][v2]):
                dist += 1
    return dist


# Return the adjacency precision and recall
# edgeset1: the true edgeset, edgeset2: the output of the algorithm
# Adjacency precision: fraction of pairs of variables adjacent in edgeset2 that are also adjacent in edgeset1
# Adjacency recall: fraction of pairs of variables adjacent in edgeset1 that are also adjacent in edgeset2
def adjacency_pr(edgeset1, edgeset2):
    adjmat1 = edgeset_to_adjmat(edgeset1)
    adjmat2 = edgeset_to_adjmat(edgeset2)

    if list(adjmat1.columns) != list(adjmat2.columns):
        # raise Exception("Debug statement: vertices are different.")
        # Add each other's missing columns for comparison
        for col in np.setdiff1d(adjmat1.columns, adjmat2.columns):
            adjmat2[col] = 0
        for col in np.setdiff1d(adjmat2.columns, adjmat1.columns):
            adjmat1[col] = 0

        # Reorder the columns
        all_cols = np.union1d(adjmat1.columns, adjmat2.columns)
        adjmat1 = adjmat1[all_cols]
        adjmat2 = adjmat2[all_cols]

        # Add each other's missing rows for comparison
        for col in np.setdiff1d(list(adjmat1.index), list(adjmat2.index)):
            adjmat2.loc[col] = 0
        for col in np.setdiff1d(list(adjmat2.index), list(adjmat1.index)):
            adjmat1.loc[col] = 0

        # Reorder the rows
        all_rows = np.union1d(list(adjmat1.index), list(adjmat2.index))
        adjmat1 = adjmat1.reindex(all_rows)
        adjmat2 = adjmat2.reindex(all_rows)

    # Make it binary since only care whether there's any edge between 2 variables
    adjmat1 = adjmat1 > 0
    adjmat2 = adjmat2 > 0

    precision_count = [0, 0]  # #pairs of variables adjacent in edgeset2, #pairs of former + also adjacent in edgeset1
    recall_count = [0, 0]  # #pairs of variables adjacent in edgeset1, #pairs of former + also adjacent in edgeset2
    for i in range(len(adjmat1.columns)):
        for j in range(i):
            v1 = adjmat1.columns[i]
            v2 = adjmat1.columns[j]

            if adjmat2[v1][v2]:
                precision_count[0] += 1
                if adjmat1[v1][v2]:
                    precision_count[1] += 1

            if adjmat1[v1][v2]:
                recall_count[0] += 1
                if adjmat2[v1][v2]:
                    recall_count[1] += 1

    precision = None if precision_count[0] == 0 else precision_count[1]/precision_count[0]
    recall = None if recall_count[0] == 0 else recall_count[1] / recall_count[0]
    return precision, recall


def discretize_imputed_df(df_imputed, df_orig, types_dict):
    df_new = df_imputed.copy()
    for i, column in enumerate(df_imputed.columns):
        print(column)
        find_dic = [types_dict[kk] for kk in range(len(types_dict)) if types_dict[kk]['name'] == column]
        if len(find_dic) > 0:
            dic = find_dic[0]
            if dic['type'] in ['bin', 'ord', 'cat']:  # if this column needs discretization
                imputed_col = np.array(df_new[dic['name']])
                orig_col = np.array(df_orig[dic['name']])
                where_nan = np.where(np.isnan(orig_col))
                where_not_nan = np.where(np.logical_not(np.isnan(orig_col)))
                if len(where_nan[0]) > 0:  # if there were nans that were imputed
                    possible_values = np.unique(imputed_col[where_not_nan])
                    if not set(np.arange(max(int(dic['nclass']), 2))).issuperset(possible_values):
                        print(possible_values)
                    assert set(np.arange(max(int(dic['nclass']), 2))).issuperset(possible_values)
                    # print("passed")
                    try:
                        closest_idxs = np.argmin((imputed_col.reshape((-1, 1)) - possible_values.reshape((1, -1))) ** 2,
                                                 axis=1)  # find the closest possible_value out of all previous values
                    except ValueError as e:
                        print(e)
                        print(dic['name'])
                        print(imputed_col)
                        print(possible_values)
                        raise Exception("{} probably completely missing. Lower missing_pc_thr".format(dic['name']))
                    assert np.all(imputed_col[where_not_nan] == possible_values[closest_idxs[where_not_nan]])
                    imputed_col[where_nan] = possible_values[closest_idxs[where_nan]]
                    df_new[dic['name']] = imputed_col
            elif dic['type'] in ['pos']:
                pass
            else:
                raise NotImplementedError
    return df_new


def entropy_edge_probs(edgeset):
    edgeset_histogram = parse_edges([edgeset], "prob")
    total_entropy = 0
    for pair in edgeset_histogram:
        total_entropy += entropy(list(edgeset_histogram[pair].values()))
    return total_entropy


def markov_blanket(target, edgeset, parents=['AgeContinuous', 'Sex']):
    mb_target = set()
    to_explore = set()
    adj = edgeset_to_adjmat(edgeset)

    # Add immediate neighbours
    if target in adj.columns:
        for i in adj[adj[target] > 0].index:
            mb_target.add(i)
            to_explore.add(i)

    # Add neighbours' neighbour if it contains a latent
    while len(to_explore) > 0:
        node = to_explore.pop()
        for i in adj[adj[node] > 2].index:  # latent edges are > 2
            if i not in mb_target and i not in [target] + parents:  # parents don't need to be explored further
                to_explore.add(i)
            if i != target:
                mb_target.add(i)
    return mb_target


def save_graph_from_file(edge_file, graph_file_prefix):
    # Load edges
    edges = {}
    with open(edge_file, 'r') as f:
        for line in f.readlines():
            lst = line.split(sep=": ", maxsplit=1)
            edges[lst[0]] = ast.literal_eval(lst[1])

    for edge_key in edges:
        est_graph_pydot_str = aggregate_edgesets([edges[edge_key]], return_type="pydot",
                                                 setting=0,  # 0 preserves all the passed edges
                                                 include_prob=True)
        save_graph_from_pydot_str(est_graph_pydot_str, "{}_{}_withprob".format(graph_file_prefix, edge_key))
        wo_prob_pydot_str = aggregate_edgesets([edges[edge_key]], return_type="pydot", setting=0)
        save_graph_from_pydot_str(wo_prob_pydot_str, "{}_{}_withoutprob".format(graph_file_prefix, edge_key))


def process_pipeline(df, features_dict_file, ycols, missing_pc_thr=np.inf, imputation_type='mice', k=20, verbose=True):
    with open(features_dict_file, 'r') as f:
        types_dict = [{k: v for k, v in row.items()}
                      for row in csv.DictReader(f, skipinitialspace=True)]
    columns_list = [types_dict[k]['name'] for k in range(len(types_dict))]
    df_sub = df[columns_list].astype('float')

    missing_pc = {}
    for ii, col in enumerate(df_sub.columns):
        val = (df_sub[col].isna().sum() / df_sub.shape[0]) * 100
        missing_pc[col] = val

    # this code filters out features if the missingness is above missing_pc_thr.
    select_features = [key for key in missing_pc.keys() if missing_pc[key] < missing_pc_thr]

    # Features in markov blanket of target
    features = np.setdiff1d(select_features, ycols)

    # Filter types_dict to only include features: not imputing with ycol
    types_dict_features = []
    for item in types_dict:
        if item['name'] in features:
            types_dict_features.append(item)

    df_impute = df_sub[features]
    df_processed = du.preprocess(df_impute, types_dict_features)
    df_impute.fillna(np.nan)

    # Call imputation
    df_processed_tr = df_processed[~df.is_test]
    df_processed_te = df_processed[df.is_test]
    df_impute_tr = df_impute[~df.is_test]
    df_impute_te = df_impute[df.is_test]

    if imputation_type == "mice":
        # Fit on training data
        xtr_list, imputers = impute_mice(df_processed_tr, 1, return_imputer=True)
        imputer = imputers[0]
        xtr = xtr_list[0]

        # Transform test data as well
        xte = imputer.transform(df_processed_te)
        xte = pd.DataFrame(data=xte, columns=xtr.columns)
        # Deprocess
        temp = du.deprocess(xtr, types_dict_features, sorted(df_impute_tr.columns), verbose=verbose)
        temp = discretize_imputed_df(temp, df_impute_tr, types_dict_features)
        xtr_deprocessed = temp

        temp = du.deprocess(xte, types_dict_features, sorted(df_impute_te.columns), verbose=verbose)
        temp = discretize_imputed_df(temp, df_impute_te, types_dict_features)
        xte_deprocessed = temp

    else:
        # KNN doesn't support transform
        temp, imputer = impute_knn(df_processed, k, return_imputer=True)
        temp = du.deprocess(temp, types_dict_features, sorted(df_impute.columns), verbose=verbose)
        temp = discretize_imputed_df(temp, df_impute, types_dict_features)
        xtr_deprocessed = temp[~df.is_test]
        xte_deprocessed = temp[df.is_test]

    return df_impute, xtr_deprocessed, xte_deprocessed

