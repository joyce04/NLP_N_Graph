#input_util.py
import pandas as pd
import numpy as np

def get_input_text():
    """
    retrieve table titles from csv
    """
    print('=========IMPORTING INPUT TEXT FILE================')
    data = pd.read_csv('titles.csv', delimiter='\t', error_bad_lines=True, header=None)
    data.columns = ['id', 'title']
    data.title = data.title.str.strip()
    documents = data
    documents['title'].replace('', np.nan, inplace=True)
    documents = documents.astype(str)
    print('checking if text is missing')
    print(documents.isna().any())
    return documents
