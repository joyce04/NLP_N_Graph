import db_conn
#import pymysql
import psycopg2
import pandas as pd
from fuzzywuzzy import fuzz

conn = db_conn.get_connection()
cursor = conn.cursor()#(pymysql.cursors.DictCursor)

# get drug list from dictionary table
cursor.execute('SELECT * FROM dict_extended_final')
drugs = pd.DataFrame(cursor.fetchall(), columns=['cui1', 'cui2', 'cui1_str', 'cui2_str'])
print(drugs[:5])
# similarity_gtl_80 = []
# for d in drugs.cui1_str.unique():
#     for sen in sents:
#         fuzz_ratio = fuzz.token_set_ratio(d, sen)
#         if fuzz_ratio >= 80:
#             similarity_gtl_80.append({'drug':d, 'sen':sen, 'ratio':fuzz_ratio})

from fuzzywuzzy import process
# choices = ['I love computer science', 'COMPUTER SCIENCE', 'computer programming', 'programming IT']
# process.extract('computer science', choices, limit=3, scorer=fuzz.token_set_ratio)

tables_count = 5500
retrieve_strip_html = """select strip_tags(content) as c, table_title, id
                        from article_tables
                        order by id
                        limit 1000 offset %s"""

possible_drugs = []
for c in range(55):
    print(c*1000)
    print(retrieve_strip_html % (c*1000))
    cursor.execute(retrieve_strip_html % (c*1000))
    result = list(cursor.fetchall())

    for d in drugs.cui1_str.unique():
    #     cursor.execute(check_query % ("'%"+d[:-cut_len]+"%'", "'%"+d+"%'"))
        fuzz_r = 0
        for r in result:
            rc = r[0].replace('\n', ' ')
            fuzz_r = fuzz.partial_token_set_ratio(d, rc)
            if fuzz_r >= 80 and fuzz_r<100:
                print(d, r[1].encode('utf-8').strip())
                words = list(filter(lambda x:x.strip() and len(x.strip())>5, rc.split(' ')))
                best_words = process.extractBests(d, words, limit=2, scorer=fuzz.token_set_ratio)
                p_drugs = list(filter(lambda x: x[1]>=70, best_words))
                print('\t' + str(p_drugs))
                possible_drugs.extend(list(map(lambda x:{'drug':d, 'p_drug':x[0], 'id':r[2], 'ratio':x[1]}, p_drugs)))

df_pos_dr = pd.DataFrame(possible_drugs)

df_pos_dr.drop_duplicates(subset=['p_drug'], inplace=True)
df_pos_dr.shape

not_in_dict = []
for dr in df_pos_dr.iterrows():

    dr_ = '%'+dr[1]['p_drug'].lower().strip().replace('(', '').replace(',', '').replace(':', '').replace('+', '').replace(';', '').replace('.', '').replace(')', '')+'%'
    cursor.execute("select * from dict_collapsed_final where lower(cui1_str) like %s or lower(cui2_str) like %s" % (dr_, dr_))
    already_in = cursor.fetchall()
    if len(already_in) ==0 and dr_.replace('%', '') not in ['otherwise', 'distribution', 'fathers', 'other','maintenance', 'father', 'dosing' ,'maintenance']:
        try:
            print(dr_.replace('%', ''))
            not_in_dict.append({'drug':dr_.replace('%', '').strip(), 'id':dr[1]['id']})
        except UnicodeDecodeError:
            print(dr_.replace("\u000B", "").replace('%', ''))
            not_in_dict.append({'drug':dr_.replace("\u000B", "").replace('%', '').strip(), 'id':dr[1]['id']})

with open('possible_drugs.tsv', 'w') as file:
    for d in not_in_dict:
        file.write(d[0]+'\t'+d[1])
