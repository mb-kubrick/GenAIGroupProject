import sqlite3
import pandas as pd
from sqlalchemy import create_engine

db_conn = sqlite3.connect('portfolio_allocations.db') 
db_cur = db_conn.cursor()

db_cur.execute(f'''
            DROP TABLE portfolio
            ''')

db_cur.execute(f'''
            CREATE TABLE IF NOT EXISTS portfolio (
            ClientId INTEGER PRIMARY KEY,
            AAPL DECIMAL(2,2) NOT NULL,
            MSFT DECIMAL(2,2),
            NVIDIA DECIMAL(2,2)
            )
            ''')

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

db_cur.execute('SELECT * FROM portfolio')
'''
rows = db_cur.fetchall()
for row in rows:
    print(rows)
'''

engine = create_engine('sqlite:///portfolio_allocations.db').connect()
df = pd.read_sql_table('portfolio', engine)
print(df)
db_conn.commit()
db_conn.close()