import pandas as pd
import swifter
import networkx as nx
import operator
from random import *

# greedy approach : select document that contains the most number of frequent entities
def find_most_effective_recursively(copy_df, ordered_vals, i):
#     print(ordered_vals[i])
    found = copy_df.loc[copy_df.val==ordered_vals[i]]
#     print(found)
    if len(found.id.unique()) > 1:
        return find_most_effective_recursively(copy_df.loc[copy_df.id.isin(found.id.unique())], ordered_vals, i+1)
    elif len(found.id.unique()) ==1:
        return found.iloc[0]
    else:
#         print(type(copy_df.groupby(copy_df.id).count().reset_index().sort_values(['val'], ascending=False).loc[0]))
        return copy_df.groupby(copy_df.id).count().reset_index().sort_values(['val'], ascending=False).loc[0]


def iterate_greedy(ordered_vals, _copy_pro_entities):

    already_annotated = []
    iter_no = 0
    
    # 제일 빈도수 높은 entity를 많이 포함하고 있는 문서 부터 annotate
    while(len(_copy_pro_entities)>0):
        val = ordered_vals[0]

        found_df = find_most_effective_recursively(_copy_pro_entities, ordered_vals, 0)
        found_id = found_df.id
        
        annotated_keywords = _copy_pro_entities.loc[_copy_pro_entities.id.eq(found_id)].val.unique()
        if len(annotated_keywords)==0:
            annotated_keywords.remove(val)
            continue

        already_annotated.append({'id': found_id, 
                                  'keywords': annotated_keywords})

        _copy_pro_entities = _copy_pro_entities.loc[(_copy_pro_entities.id!=found_id) & (~_copy_pro_entities.val.isin(annotated_keywords))]
        _copy_pro_entities.sort_index(inplace=True)

        for an in annotated_keywords:
            ordered_vals.remove(an)
    #     print(len(ordered_vals))
        iter_no += 1

#     print(iter_no)
    return iter_no

## pagerank with random walk
def generate_edge_weights(_df_pro_entities):
    p_edges = {}
    for val, grp in _df_pro_entities.groupby('val'):
        if len(grp) > 1:
            id_list = grp.id.unique()
            for i in range(len(id_list)-1):
                if id_list[i]+'|'+id_list[i+1] not in p_edges.keys():
                    p_edges[id_list[i]+'|'+id_list[i+1]] = 1
                else:
                    p_edges[id_list[i]+'|'+id_list[i+1]] += 1

                if id_list[i+1]+'|'+id_list[i] not in p_edges.keys():
                    p_edges[id_list[i+1]+'|'+id_list[i]] = 1
                else:
                    p_edges[id_list[i+1]+'|'+id_list[i]] += 1
    return p_edges

def build_graph(_df_pro_entities):
    G = nx.DiGraph()

    for k in _df_pro_entities.id.unique():
        G.add_node(k)
    
    p_edges = generate_edge_weights(_df_pro_entities)
    for key, val in p_edges.items():
    #     print(key)
        keys = key.split('|')
        G.add_edge(keys[0], keys[1], weight=val)

    return G
    
def compute_pagerank(_df_pro_entities):
    G = build_graph(_df_pro_entities)
    pageranked_list = nx.pagerank(G, alpha=0.0001)
#     max(pageranked_list.items(), key=operator.itemgetter(1))
    return pageranked_list

def iterate_pagerank(_ordered_vals, _df_pro_entities):
    pageranked_list = compute_pagerank(_df_pro_entities)

    sort_ranked = list(map(lambda x: x[0], sorted(pageranked_list.items(), key=operator.itemgetter(1), reverse=True)))
    
    copy_pro_entities = _df_pro_entities.copy(deep=True)

    already_annotated = []
    iter_no = 0
    i = 0

    while(len(copy_pro_entities)>0):
        found_id = sort_ranked[0]

        if len(copy_pro_entities.loc[copy_pro_entities.id.eq(found_id)])==0:
            sort_ranked.remove(found_id)
            continue
        annotated_keywords = copy_pro_entities.loc[copy_pro_entities.id.eq(found_id)].val.unique()        
        already_annotated.append({'id': found_id, 
                                  'keywords': annotated_keywords})

        copy_pro_entities = copy_pro_entities.loc[(copy_pro_entities.id!=found_id) & (~copy_pro_entities.val.isin(annotated_keywords))]
        copy_pro_entities.sort_index(inplace=True)
        sort_ranked.remove(found_id)

        for an in annotated_keywords:
            _ordered_vals.remove(an)

        iter_no += 1
#     print(iter_no)
    return iter_no

## random order
def iterate_random(_ordered_vals, _df_pro_entities, _id_list):
        copy_pro_entities = _df_pro_entities.copy(deep=True)

        already_annotated = []
        iter_no = 0
        while(len(copy_pro_entities)>0):
            ran_id = choice(_id_list)
            _id_list.remove(ran_id)

            if len(copy_pro_entities.loc[copy_pro_entities.id.eq(ran_id)])==0:
                continue

            annotated_keywords = copy_pro_entities.loc[copy_pro_entities.id.eq(ran_id)].val.unique()
            already_annotated.append({'id': ran_id, 
                                      'keywords': annotated_keywords})

            copy_pro_entities = copy_pro_entities.loc[(copy_pro_entities.id!=ran_id) & (~copy_pro_entities.val.isin(annotated_keywords))]
            copy_pro_entities.sort_index(inplace=True)

            for an in annotated_keywords:
                _ordered_vals.remove(an)
            iter_no += 1
        
#         print(iter_no)
        return iter_no
