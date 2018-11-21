import pymysql

def connection(host='madigun.asuscomm.com', user='madigun', password='ehenr1163', db='football_worldcup'):
    global CONN
    # MySQL DB Connection
    CONN = pymysql.connect(host='madigun.asuscomm.com', user='madigun', password='ehenr1163', db='research', charset='utf8')


def select_query(sql):
    # SQL 실행
    curs = CONN.cursor(pymysql.cursors.DictCursor)
    curs.execute(sql)

    result = curs.fetchall()
    return result

def close():
    CONN.close()

connection()