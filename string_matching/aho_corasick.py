#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 7 13:36:05 2018

@author: grace
"""

import queue


#Aho-corasick
"""
preprocessing: goto, failure, output
 goto: 2D array g[][] where next state for current state and character
 failure: 1D array f[] where next state for current state when current character doesn't have edge in Trie
 ouput: 1D array o[] indexes of all matching patterns for current state

input : patterns
output: 
"""
def build_matching_machine(patterns):
    # maximum number of states
    MAX_LENGTH = 1000
    # maximum number of characters in input alphabet
    MAX_C = 300
    
    out = [0]*MAX_LENGTH
    failure = [-1]*MAX_LENGTH
    goto = [[-1]*MAX_C for _ in range(MAX_LENGTH)]
    result = dict()


    # initialize state
    state = 1
    
    # goto function
    for i, pattern in enumerate(patterns):
        result[pattern] = []
        current_state = 0
        
        # insert all characters of current word
        # save ord value from character
        for c in pattern:
            ch = ord(c) - ord(' ')
            print(c, ch)
            
            #new state of a node for ch doesn't exit
            if (goto[current_state][ch] == -1):
                goto[current_state][ch] = state
                state = state + 1
            
            current_state = goto[current_state][ch]
            
        #add current word in output function
        # use binary format to include multiple outputs
        out[current_state] = out[current_state] | (1 << i)
    
    # for characters which don't have an edge from root
    # add goto edge to state 0
    for ch in range(MAX_C):
        if goto[0][ch] == -1:
            goto[0][ch] = 0
            
    # failure function
    # using BFS
    q = queue.Queue()
    
    # all possible input
    for ch in range(MAX_C):
        # all nodes of dept 1 have failure function as 0
        if goto[0][ch] != 0:
            failure[goto[0][ch]] = 0
            q.put(goto[0][ch])
#            print(goto[0][ch])
    
    while(q.qsize()):
#        print(q.qsize())
        new_state = q.get()
        # compute failure function for poped state
        for ch in range(MAX_C):
            if goto[new_state][ch] != -1:
                fail = failure[new_state]
                
                while (goto[fail][ch] == -1):
                    fail = failure[fail]
                fail = goto[fail][ch]
                failure[goto[new_state][ch]] = fail
                
                out[goto[new_state][ch]] = out[goto[new_state][ch]] | out[fail]
                q.put(goto[new_state][ch])

    return goto, failure, out, result                    

def get_next_state(next_ch, current_state, goto, failure):
    ch = ord(next_ch) - ord(' ')
    
    follow_state = current_state
    # follow failure fuction
    while goto[follow_state][ch] == -1:
        follow_state = failure[follow_state]
    return goto[follow_state][ch]

def search(text, patterns):
    """
#        >>> aho_corasick_search('bacabcabcabca', ['abcabca'])
#        ({'abcabca': [3, 6]}, [0, 1, 0, 1, 2, 3, 4, 5, 6, 7, 5, 6, 7])
#        >>> aho_corasick_search('ahishers', ['he', 'she', 'hers', 'his'])
#        ({'he': [4], 'she': [3], 'hers': [4], 'his': [1]}, [0, 1, 8, 9, 4, 5, 6, 7])
    """
    # initalize
    bb_out = []

    goto, failure, out, result = build_matching_machine(patterns)
    current_state = 0
    
    for i, t in enumerate(text):
        current_state = get_next_state(t, current_state, goto, failure)
        bb_out.append(current_state)
        
        if out[current_state] != 0:
            for j in range(len(patterns)+1):
                if out[current_state] & (1 << j):
                    print('%s appears in at %s' % (patterns[j], str(i-len(patterns[j])+1)))
                    result[patterns[j]].append(i-len(patterns[j])+1)
#    return bb_out
    return result

#search('bacabcabcabca', ['abcabca'])
#search("antihypertension treatment 69 (54%) , subclinical cushing's syndrome (n=10) : 0·13", ['hypertension', 'anti'])