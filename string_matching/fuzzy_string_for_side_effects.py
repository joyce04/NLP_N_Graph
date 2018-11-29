import db_conn
#import pymysql
import psycopg2
import pandas as pd
from fuzzywuzzy import fuzz
import spacy
import en_core_web_sm

nlp = en_core_web_sm.load()
conn = db_conn.get_connection()
cursor = conn.cursor()#(pymysql.cursors.DictCursor)

# get drug list from dictionary table
cursor.execute('SELECT llt_code, llt_name FROM meddra_llt_181022')
side_effects = pd.DataFrame(cursor.fetchall(), columns=['llt_code', 'llt_name'])
print(side_effects[:5])

from fuzzywuzzy import process
tables_count = 5500
retrieve_strip_html = """select strip_tags(content) as c, table_title, id
                        from article_tables
                        order by id
                        limit 1000 offset %s"""

possible_side_effects = []
not_in_dict = []
for c in range(55):
    print(c*1000)
    # print(retrieve_strip_html % (c*1000))
    cursor.execute(retrieve_strip_html % (c*1000))
    result = list(cursor.fetchall())

    for d in side_effects.llt_name.unique():
        fuzz_r = 0
        for r in result:
            rc = r[0].replace('\n', ' ')
            fuzz_r = fuzz.partial_token_set_ratio(d, rc)
            if fuzz_r >= 80 and fuzz_r<100:
                print(d, r[1].encode('utf-8').strip())
                doc = nlp(rc)
                n_chunks = [chunk for chunk in doc.noun_chunks]
                best_words = process.extractBests(d, n_chunks, limit=2, scorer=fuzz.token_set_ratio)
                p_drugs = list(map(lambda x: str(str(x[0]).replace('\u2217', '').replace('\xb1', '')), list(filter(lambda x: x[1]>=70, best_words))))
                p_drugs = list(filter(lambda x: lower(x) not in ['group', 'groups', 'expression', 'change', 'age', 'stranger', 'range'], p_drugs))
                for p in p_drugs:
                    print('\t' + str(p_drugs))
                    dr_ = "'%"+p.lower().strip().replace('(', '').replace(',', '').replace(':', '').replace('+', '').replace(';', '').replace('.', '').replace(')', '')+"%'"
                    cursor.execute("select * from meddra_llt_181022 where lower(llt_name) like %s" % (dr_, ))
                    already_in = cursor.fetchall()
                    if len(already_in) ==0:
                        try:
                            print(dr_.replace('%', ''))
                            not_in_dict.append({'p_s':dr_.replace('%', '').strip(), 'id':r[2]})
                        except UnicodeDecodeError:
                            print(dr_.replace("\u000B", "").replace('%', ''))
                            not_in_dict.append({'p_s':dr_.replace("\u000B", "").replace('%', '').strip(), 'id':r[2]})

not_in_dict_df = pd.DataFrame(not_in_dict)
not_in_dict_df.drop_duplicates(subset=['p_s'], inplace=True)

for n in not_in_dict.p_s.unique():
    print(n)
with open('possible_side_effects.tsv', 'w') as file:
    for d in not_in_dict.iterrows():
        file.write(d[0]+'\t'+d[1])
