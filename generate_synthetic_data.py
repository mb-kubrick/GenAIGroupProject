import sqlite3
import pandas as pd
from sqlalchemy import create_engine

db_conn = sqlite3.connect('portfolio_allocations.db') 
db_cur = db_conn.cursor()
#db_name = 'portfolio_allocations.db'

'''
def get_connections(db_name):
    db_conn = sqlite3.connect(db_name) 
    db_cur = db_conn.cursor()
    return db_conn, db_cur

db_conn = get_connections(db_name)[0]
db_cur = get_connections(db_name)[1]
print(db_conn)
print(db_cur)
'''

def drop_table(db_conn, db_cur):
    try:
        db_cur.execute(f'''
            DROP TABLE portfolio
            ''')
        db_conn.commit()
        print(f' table portfolio dropped')
    except Exception as e:
        print(f'drop_table: {e}')


def create_table(db_conn, db_cur):
    try:
        db_cur.execute(f'''
            CREATE TABLE IF NOT EXISTS portfolio (
            ClientId INTEGER PRIMARY KEY,
            AAPL DECIMAL(2,2) NOT NULL,
            MSFT DECIMAL(2,2),
            NVIDIA DECIMAL(2,2)
            )
            ''')
        db_conn.commit()
        print(f'table portfolio created')
    except Exception as e:
        print(e)

def insert_into_table(db_conn, db_cur):
    try:
        db_cur.execute('''
            INSERT INTO portfolio (
            ClientID, AAPL, MSFT, NVIDIA)
            VALUES 
            (1, 0.40, 0.20, 0.40),
            (2, 0.25, 0.25, 0.5),
            (3, 0.25, 0.25, 0.5),
            (4, 0.30, 0.35, 0.35),
            (5, 0.05, 0.05, 0.90),
            (6, 0.15, 0.65, 0.20),
            (7, 0.20, 0.20, 0.60),
            (8, 0.30, 0.25, 0.45),
            (9, 0.10, 0.80, 0.10),
            (10, 0.45, 0.20, 0.35)''')

        db_conn.commit()
        print(f'insert into portfolio successful')
    except Exception as e:
        print(e)  


drop_table(db_conn, db_cur)
create_table(db_conn, db_cur)
insert_into_table(db_conn, db_cur)
#db_conn.commit()

'''
db_cur.execute('SELECT * FROM portfolio')
rows = db_cur.fetchall()
for row in rows:
    print(rows)
'''

engine = create_engine('sqlite:///portfolio_allocations.db').connect()
df = pd.read_sql_table('portfolio', engine)
print(df)
db_conn.commit()
db_conn.close()

