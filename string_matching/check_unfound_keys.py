import math
from db_conn import get_connection
import datetime
from aho_corasick import search

def get_side_effects():
    _cur = conn.cursor()
    select_sql = """select distinct(lower(llt_name)) from meddra_llt_180717 WHERE exclude=0 order by lower(llt_name)"""
    _cur.execute(select_sql)
    return _cur.fetchall()

def get_drugs():
    _cur = conn.cursor()
    select_sql = """select distinct(lower(cui1_str)) from dict_collapsed_final order by lower(cui1_str)"""
    _cur.execute(select_sql)
    return _cur.fetchall()

def get_dr_keywords():
    drugs = list(map(lambda x: x[0], get_drugs()))
    return drugs

def get_sd_keywords():
    side_effects = list(map(lambda x: x[0], get_side_effects()))
    return side_effects

def copy_into_table_m(col, rows):
    _cur = conn.cursor()
    if col=='drug':
        _cur.executemany(
            '''
                UPDATE article_table_sentences_m
                SET
                    check_drug = %(drug)s
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
                    check_ad = %(adverse_effect)s
                WHERE
                    id = %(id)s
            ''',
            tuple(rows)
        )

    row_count = _cur.rowcount
    conn.commit()
    print(row_count)

def update_check_col(col, _id, name):
    print(_id, name)
    if col == 'check_drug':
        update_sql = """
            UPDATE article_table_sentences_m
            SET check_drug = %s
            WHERE id = %s
        """
    else:
        update_sql = """
            UPDATE article_table_sentences_m
            SET check_ad = %s
            WHERE id = %s
        """

    _cur = conn.cursor()
    _cur.execute(update_sql, (name, _id))
    conn.commit()
    # print(_cur.rowcount)


def get_unserached_entities(col, keywords):
    file = open(col+'_check.tsv', 'w')

    _cur = conn.cursor()

    for k in keywords:
        print(k)
        if col=='check_drug':
            _cur.execute(
                """
                    SELECT id, sentence, check_drug
                    FROM article_table_sentences_m
                    WHERE lower(sentence) like %s
                    AND lower(drug) not like %s
                """, ("%"+k.strip()+"%", "%"+k.strip()+"%")
            )
            # print(cursor.mogrify(SQL, ("%"+k.strip()+"%", "%"+k.strip()+"%")))
        else:
            _cur.execute(
            """
                SELECT id, sentence, check_ad
                FROM article_table_sentences_m
                WHERE lower(sentence) like %s
                AND lower(adverse_effect) not like %s
            """, ("%"+k.strip()+"%", "%"+k.strip()+"%")
            )

        fetched = _cur.fetchall()
        # print(fetched)

        ch_founds = [k]
        update_list_m = []
        if len(fetched)>0:
            print(len(fetched))
            # print(fetched[0])
            for a in fetched:
                s = a[1].strip().lower().replace('=', ' = ').replace(']', ' ] ')
                # print(s)
                front_end = s.split(k)
                # print(front_end)
                a_loc = s.find(k)
                if len(front_end)>1:
                    front_space = front_end[0].rfind(' ')
                    # print(front_space)
                    end_space = front_end[1].find(' ')
                    # print(end_space)
                else:
                    if len(s) - a_loc == len(a):
                        front_space = front_end[0].rfind(' ')
                        end_space = a_loc+len(a)
                    else:
                        end_space = front_end[0].find(' ')
                        front_space = a_loc
                # print(s[front_space:a_loc+len(k)+end_space+1])
                name = s[front_space:a_loc+len(k)+end_space+1].strip()
                if name == k.strip():
                    if a[2] is not None:
                        name = a[2]+' , '+name
                    update_check_col(col, a[0], name)
                else:
                    replace_phrases = ['-controlled', '-treated', '-based', '-related', '-releasing']
                    name = name.replace(')', '').replace(':', '').replace('(', '').replace('+', '').replace('*', '').replace('[', '').replace('.', '').replace(',', '').replace('%', '')
                    if name.strip():
                        if a[2] is not None:
                            name = a[2]+' , '+name
                        update_check_col(col, a[0], name)
                ch_founds.append(name)
                # update_list_m.append({'id':a[0], col:a[2]+' , '+name})

        # if len(update_list_m)>0:
        #     copy_into_table_m('drug', update_list_m)

        if len(ch_founds)>1:
            ch_founds = list(set(filter(lambda x: x.strip(), ch_founds)))

            if len(ch_founds)>1:
                # ch_founds = list(filter(lambda x: x.strip() != k.strip(), ch_founds))
                ch_founds.remove(k.strip())
                print(ch_founds)
                file.write(k+'\t'+' , '.join(ch_founds)+'\n')
    file.close()

conn = get_connection()
drugs = get_dr_keywords()
get_unserached_entities('check_drug', drugs)
# sd = get_sd_keywords()
# get_unserached_entities('check_drug_ad', sd)
