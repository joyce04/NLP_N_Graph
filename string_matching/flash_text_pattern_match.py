import math
from flashtext import KeywordProcessor
from db_conn import get_connection
import datetime
from aho_corasick import search

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

# def get_sentences_to_search(ids):
#     _cur = conn.cursor()
#     select_sql = """select id, sentence from article_table_sentences where table_id in %s"""
#
#     _cur.execute(select_sql, (tuple(ids),))
#     row_count = _cur.rowcount
#     return _cur.fetchall()

def get_sentences_to_search_m(ids):
    _cur = conn.cursor()
    select_sql_m = """select id, sentence from article_table_sentences_m where table_id in %s"""

    _cur.execute(select_sql_m, (tuple(ids),))
    row_count = _cur.rowcount

    return _cur.fetchall()

# def copy_into_table(col, rows):
#     _cur = conn.cursor()
#     if col =='drug':
#         _cur.executemany(
#             '''
#                 UPDATE article_table_sentences
#                 SET
#                     drug = %(drug)s
#                 WHERE
#                     id = %(id)s
#             ''',
#             tuple(rows)
#         )
#     else:
#         _cur.executemany(
#             '''
#                 UPDATE article_table_sentences
#                 SET
#                     adverse_effect = %(adverse_effect)s
#                 WHERE
#                     id = %(id)s
#             ''',
#             tuple(rows)
#         )
#
#     row_count = _cur.rowcount
#     conn.commit()
#     print(row_count)

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

def get_dr_keywords():
    drugs = list(map(lambda x: x[0], get_drugs()))
    keyword_processor = KeywordProcessor(case_sensitive=False)
    for d in drugs:
        d = d.encode('ascii', 'replace').decode('ascii')
        d = d.replace('?', ' ')
        keyword_processor.add_keyword((' '+d.strip()))#.encode('ascii', 'replace'))
        keyword_processor.add_keyword((d.strip()+' '))#.encode('ascii', 'replace'))
        keyword_processor.add_keyword((' '+d.strip()+' '))#.encode('ascii', 'replace'))
        keyword_processor.add_keyword(('-'+d.strip()))#.encode('ascii', 'replace'))
        keyword_processor.add_keyword((d.strip()+'-'))#.encode('ascii', 'replace'))
    return keyword_processor, drugs

def update_drugs(ids, keyword_processor, drugs):
    # update_list = []
    # sentences = get_sentences_to_search(ids)
    # for sen in sentences:
    #     id = sen[0]
    #
    #     ori_found = []
    #     found = []
    #     found = keyword_processor.extract_keywords(sen[1].encode('ascii', 'replace').strip().lower())
    #     if len(found)>0:
    #         _found = list(set(list(map(lambda x: x.decode('utf-8'), found))))
    #         sen_strs = sen[1].encode('ascii', 'replace').strip().lower().split(' ')
    #         print(_found)
    #         print(sen[1].encode('ascii', 'replace').strip().lower())
    #         for f in _found:
    #             ori_found.extend(list(filter(lambda x: x.find(f)>=0, sen_strs)))
    #         update_list.append({'id':id, 'drug':' '+' , '.join(list(set(ori_found)))+' '})
    #
    # if len(update_list)>0:
    #     print('update')
    #     print(update_list)
    #     copy_into_table('drug', update_list)

    update_list_m = []
    sentences_m = get_sentences_to_search_m(ids)
    for sen in sentences_m:
        id = sen[0]

        ori_found = []
        found = []
        s = (u' '+sen[1].lower()).encode('ascii', 'replace').decode('ascii')
        s = str(s).replace('?', ' ').replace(';', ' ; ').replace('/', ' / ').replace('=', ' = ')
        print(s)
        found = keyword_processor.extract_keywords(s)
        if len(found)>0:
            # _found = list(map(lambda x: x.decode('utf-8'), found))
            _found = found#list(map(lambda x: x, found))
            sen_strs = s.split(' ')
            # print(sen_strs)
            print(s)
            print('FOUND:   ', _found)

            for f in _found:
                ori_found.extend(list(filter(lambda x: (' '+x+' ').find(f)>=0, sen_strs)))
            # print('ORIGIN:   ', ori_found)
            ori_found = check_duplicates(_found, ori_found)
            # ori_found = list(set(map(lambda x: str(x).strip(), ori_found)))
            # update_list_m.append({'id':id, 'drug':' '+' , '.join(ori_found)+' '})

        # aho_founds = []
        # for i in range(math.ceil(len(drugs)/200)):
        #     a_f = search(s, drugs[200*i:200*(i+1)])
        #     if len(a_f)>0:
        #         # for a in a_f:
        #         #     front_end = s.split(a)
        #         #     if len(front_end)>1:
        #         #         front_space = front_end[0].rfind(' ')
        #         #         end_space = front_end[1].find(' ')
        #         #     else:
        #         #         a_loc = s.find(a)
        #         #         if len(s) - a_loc == len(a):
        #         #             front_space = front_end[0].rfind(' ')
        #         #             end_space = a_loc+len(a)
        #         #         else:
        #         #             end_space = front_end[0].find(' ')
        #         #             front_space = a_loc
        #         #     aho_founds.append(s[front_space:end_space+1])
        #         aho_founds.extend(a_f)
        #
        # if len(aho_founds)>0:
        #     print(aho_founds)
        #     ori_found = check_duplicates(aho_founds, ori_found)

        if len(ori_found)>0:
            ori_found = list(set(map(lambda x: str(x).strip(), ori_found)))
            print('ORIGIN:   ', ori_found)
            update_list_m.append({'id':id, 'drug':' '+' , '.join(ori_found)+' '})

    if len(update_list_m)>0:
        print('update m')
        copy_into_table_m('drug', update_list_m)

def check_duplicates(first_f, origin_f):
    origin_f = list(map(lambda x: x.strip().replace(')', '').replace(':', '').replace('(', '').replace('+', '').replace('*', '').replace('[', ''), origin_f))
    single_words_f = list(filter(lambda x: x.strip().find(' ')<0, first_f))
    for s in single_words_f:
        if s in origin_f:
            first_f.remove(s)
    origin_f.extend(first_f)
    print(origin_f)
    return origin_f

def get_sd_keywords():
    side_effects = list(map(lambda x: x[0], get_side_effects()))
    keyword_processor = KeywordProcessor(case_sensitive=False)
    for side in side_effects:
        side = side.encode('ascii', 'replace').decode('ascii')
        side = side.replace('?', ' ')
        keyword_processor.add_keyword((' '+side.strip()))#.encode('ascii', 'replace'))
        keyword_processor.add_keyword((side.strip()+' '))#.encode('ascii', 'replace'))
        keyword_processor.add_keyword((' '+side.strip()+' '))#.encode('ascii', 'replace'))
        keyword_processor.add_keyword(('-'+side.strip()))#.encode('ascii', 'replace'))
        keyword_processor.add_keyword((side.strip()+'-'))#.encode('ascii', 'replace'))
    return keyword_processor, side_effects

def update_llts(ids, keyword_processor, side_effects):
#     update_list = []
#     sentences = get_sentences_to_search(ids)
#     for sen in sentences:
#         id = sen[0]

#         found = []
#         found = keyword_processor.extract_keywords(sen[1].encode('ascii', 'replace').strip().lower())
#         if len(found)>0:
#             _found = list(map(lambda x: x.decode('utf-8'), found))
#             sen_strs = sen[1].encode('ascii', 'replace').strip().lower().split(' ')
#             print(_found)
#             print(sen[1].encode('ascii', 'replace').strip().lower())
#             update_list.append({'id':id, 'adverse_effect':' '+' , '.join(_found)+' '})

#     if len(update_list)>0:
#         print('update')
#         print(update_list)
#         copy_into_table('ad', update_list)

    update_list_m = []
    sentences_m = get_sentences_to_search_m(ids)
    for sen in sentences_m:
        id = sen[0]

        ori_found = []
        found = []
        s = (u' '+sen[1].lower()).encode('ascii', 'replace').decode('ascii')
        s = str(s).replace('?', ' ').replace(';', ' ; ').replace('/', ' / ').replace('=', ' = ')
        print(s)
        found = keyword_processor.extract_keywords(s)
        if len(found)> 0:
            # _found = list(map(lambda x: x.decode('utf-8'), found))
            _found = found
            sen_strs = s.split(' ')
            # print(sen_strs)
            # print(s)
            print('FOUND:   ', _found)

            for f in _found:
                ori_found.extend(list(filter(lambda x: (' '+x+' ').find(f)>=0, sen_strs)))
            # print('ORIGIN:   ', ori_found)
            ori_found = check_duplicates(_found, ori_found)

        # print(side_effects[:10])
        # aho_founds = []
        # for i in range(math.ceil(len(side_effects)/200)):
        #     a_f = search(s, side_effects[200*i:200*(i+1)])
        #     if len(a_f)>0:
        #         aho_founds.extend(a_f)
        # if len(aho_founds)>0:
        #     print(aho_founds)
        #     ori_found = check_duplicates(aho_founds, ori_found)

        if len(ori_found)>0:
            ori_found = list(set(map(lambda x: str(x).strip(), ori_found)))
            print('ORIGIN:   ', ori_found)
            update_list_m.append({'id':id, 'adverse_effect':' '+' , '.join(ori_found)+' '})


    if len(update_list_m)>0:
        print('update m')
        copy_into_table_m('ad', update_list_m)



dr_keyword_processor, drugs = get_dr_keywords()
print('drugs:', len(drugs))
sd_keyword_processor, sd = get_sd_keywords()
print('sd:', len(sd))
for i in range(1000, len(ids)+1000, 1000):
    print(str(i-1000)+':'+str(i))
    print('llts : ' +str(update_llts(ids[i-1000:i], sd_keyword_processor, sd)))
    print('drugs:' + str(update_drugs(ids[i-1000:i], dr_keyword_processor, drugs)))

print(start_time - datetime.datetime.now())
