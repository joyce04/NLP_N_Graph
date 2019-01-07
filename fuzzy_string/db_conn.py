import psycopg2

psycopg2.extensions.register_type(psycopg2.extensions.UNICODEARRAY)
psycopg2.extensions.register_type(psycopg2.extensions.UNICODE)

def get_connection():
    conn = psycopg2.connect(host='smith.snu.ac.kr', database='daeb', user='daeb', password='daeb123123')
    return conn

conn = get_connection()
conn.set_client_encoding('UTF8')