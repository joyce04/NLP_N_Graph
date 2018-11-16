import math
from flashtext import KeywordProcessor
from db_conn import get_connection
import datetime

def get_table_ids():
    _cur = conn.cursor()
    select_sql = """select id from article_tables order by id"""
    _cur.execute(select_sql)
    return _cur.fetchall()

def get_side_effects():
    _cur = conn.cursor()
    select_sql = """select distinct(lower(llt_name)) from meddra_llt_180717 WHERE exclude=0"""
    _cur.execute(select_sql)
    return _cur.fetchall()

def get_drugs():
    _cur = conn.cursor()
    select_sql = """select distinct(lower(cui1_str)) from dict_collapsed_final"""
    _cur.execute(select_sql)
    return _cur.fetchall()

def get_sentences_to_search(ids):
    _cur = conn.cursor()
    select_sql = """select id, sentence from article_table_sentences where table_id in %s"""

    _cur.execute(select_sql, (tuple(ids),))
    row_count = _cur.rowcount
    return _cur.fetchall()

def get_sentences_to_search_m(ids):
    _cur = conn.cursor()
    select_sql_m = """select id, sentence from article_table_sentences_m where table_id in %s"""

    _cur.execute(select_sql_m, (tuple(ids),))
    row_count = _cur.rowcount

    return _cur.fetchall()

def copy_into_table(col, rows):
    _cur = conn.cursor()
    if col =='drug':
        _cur.executemany(
            '''
                UPDATE article_table_sentences
                SET
                    drug = %(drug)s
                WHERE
                    id = %(id)s
            ''',
            tuple(rows)
        )
    else:
        _cur.executemany(
            '''
                UPDATE article_table_sentences
                SET
                    adverse_effect = %(adverse_effect)s
                WHERE
                    id = %(id)s
            ''',
            tuple(rows)
        )

    row_count = _cur.rowcount
    conn.commit()
    print(row_count)

def copy_into_table_m(col, rows):
    _cur = conn.cursor()
    if col=='drug':
        _cur.executemany(
            '''
                UPDATE article_table_sentences_m
                SET
                    drug = %(drug)s
                WHERE
                    id = %(id)s
            ''',
            tuple(rows)
        )
    else:
                _cur.executemany(
            '''
                UPDATE article_table_sentences_m
                SET
                    adverse_effect = %(adverse_effect)s
                WHERE
                    id = %(id)s
            ''',
            tuple(rows)
        )

    row_count = _cur.rowcount
    conn.commit()
    print(row_count)

start_time = datetime.datetime.now()

conn = get_connection()

ids = list(map(lambda x:x[0], get_table_ids()))

def update_drugs(ids):
    drugs = list(map(lambda x: x[0], get_drugs()))
    keyword_processor = KeywordProcessor(case_sensitive=False)
    for d in drugs:
        keyword_processor.add_keyword((' '+d.strip()+' ').encode('utf-8'))
        keyword_processor.add_keyword(('-'+d.strip()).encode('utf-8'))
        keyword_processor.add_keyword((d.strip()+'-').encode('utf-8'))

    update_list = []
    sentences = get_sentences_to_search(ids)
    for sen in sentences:
        id = sen[0]

        found = []
        found = keyword_processor.extract_keywords(sen[1].encode('utf-8').strip().lower())
        if len(found)>0:
            _found = list(set(list(map(lambda x: x.decode('utf-8'), found))))
            print(_found)
            print(sen[1].encode('utf-8').strip().lower())
            update_list.append({'id':id, 'drug':' '+' , '.join(_found)+' '})

    if len(update_list)>0:
        print('update')
        print(update_list)
        copy_into_table('drug', update_list)

    update_list_m = []
    sentences_m = get_sentences_to_search_m(ids)
    for sen in sentences_m:
        id = sen[0]

        found = []
        found = keyword_processor.extract_keywords(sen[1].encode('utf-8').strip().lower())
        if len(found)>0:
            _found = list(map(lambda x: x.decode('utf-8'), found))
            print(_found)
            print(sen[1].encode('utf-8').strip().lower())
            update_list_m.append({'id':id, 'drug':' '+' , '.join(_found)+' '})

    if len(update_list_m)>0:
        print('update m')
        copy_into_table_m('drug', update_list_m)

def update_llts(ids):
    side_effects = list(map(lambda x: x[0], get_side_effects()))
    keyword_processor = KeywordProcessor(case_sensitive=False)
    for side in side_effects:
        keyword_processor.add_keyword((' '+side.strip()+' ').encode('utf-8'))
        keyword_processor.add_keyword(('-'+side.strip()).encode('utf-8'))
        keyword_processor.add_keyword((side.strip()+'-').encode('utf-8'))

    update_list = []
    sentences = get_sentences_to_search(ids)
    for sen in sentences:
        id = sen[0]

        found = []
        found = keyword_processor.extract_keywords(sen[1].encode('utf-8').strip().lower())
        if len(found)>0:
            _found = list(map(lambda x: x.decode('utf-8'), found))
            print(_found)
            print(sen[1].encode('utf-8').strip().lower())
            update_list.append({'id':id, 'adverse_effect':' '+' , '.join(_found)+' '})

    if len(update_list)>0:
        print('update')
        print(update_list)
        copy_into_table('ad', update_list)

    update_list_m = []
    sentences_m = get_sentences_to_search_m(ids)
    for sen in sentences_m:
        id = sen[0]

        found = []
        found = keyword_processor.extract_keywords(sen[1].encode('utf-8').strip().lower())
        if len(found)>0:
            _found = list(map(lambda x: x.decode('utf-8'), found))
            print(_found)
            print(sen[1].encode('utf-8').strip().lower())
            update_list_m.append({'id':id, 'adverse_effect':' '+' , '.join(_found)+' '})

    if len(update_list_m)>0:
        print('update m')
        copy_into_table_m('ad', update_list_m)

for i in range(1000, len(ids)+1000, 1000):
    print(str(i-1000)+':'+str(i))
    print('llts : ' +str(update_llts(ids[i-1000:i])))
    print('drugs:' + str(update_drugs(ids[i-1000:i])))
<<<<<<< Updated upstream

print(start_time - datetime.datetime.now())
=======
    
print(start_time - datetime.datetime.now())
>>>>>>> Stashed changes
