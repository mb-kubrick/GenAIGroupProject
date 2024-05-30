import pandas as pd
import random
import sqlite3
import yfinance as yf
from sqlalchemy import create_engine

db_conn = sqlite3.connect('portfolio_allocations.db') 
db_cur = db_conn.cursor()

def gen_norm_vals():
    values = [round(random.random(), 2) for _ in range(5)]
    total = round(sum(values), 2)
    norm_vals = [round(val/ total, 2) for val in values]
    return norm_vals

def get_insert_list(num_of_clients):
    insert_list = []
    for _ in range(num_of_clients):
        values = gen_norm_vals()
        insert_list.append(values)
    return insert_list

def get_shares_list():
    shares_list = []
    for _ in range(len(df)):
        shares_list.append(random.randint(50,1000))
    return shares_list

def get_ticker_price(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period='1d')
    latest_price = data['Close'].iloc[-1]
    return latest_price

def drop_table(db_conn, db_cur, table_name):
    try:
        db_cur.execute(f'''
            DROP TABLE {table_name}
            ''')
        db_conn.commit()
        print(f' table {table_name} dropped')
    except Exception as e:
        print(f'drop_table: {e}')


def create_table(db_conn, db_cur):
    try:
        db_cur.execute(f'''
            CREATE TABLE IF NOT EXISTS portfolio (
            ClientId INTEGER PRIMARY KEY AUTOINCREMENT,
            AAPL DECIMAL(2,2) NOT NULL,
            MSFT DECIMAL(2,2),
            NVDA DECIMAL(2,2),
            F DECIMAL(2,2),
            GOOGL DECIMAL(2,2))
            ''')
        db_conn.commit()
        print(f'table portfolio created')
    except Exception as e:
        print(e)

def insert_into_table(db_conn, db_cur, val_list):
    try:
        db_cur.executemany(f'''
            INSERT INTO portfolio (
            AAPL, MSFT, NVDA, F, GOOGL)
            VALUES 
            (?,?,?,?,?)''', val_list)

        db_conn.commit()
        print(f'insert into portfolio successful')
    except Exception as e:
        print(e)  

num_of_clients = 15
insert_list =  get_insert_list(num_of_clients)

drop_table(db_conn, db_cur, 'portfolio')
create_table(db_conn, db_cur)
insert_into_table(db_conn, db_cur, insert_list)

engine = create_engine('sqlite:///portfolio_allocations.db').connect()
df = pd.read_sql_table('portfolio', engine)

db_conn.commit()
#db_conn.close()

shares_list = get_shares_list()

AAPL_price = get_ticker_price('AAPL')
MSFT_price = get_ticker_price('MSFT')
NVIDIA_price = get_ticker_price('NVDA')
F_price = get_ticker_price('F')
GOOGL_price = get_ticker_price('GOOGL')


df.insert(6, "shares held", shares_list)
#df.info()
#(68*0.128*190) + (68*0.152*492) + (68*0.720*1148) for total value 
df=df.assign(value_USD = lambda x: (round((x['AAPL']*x['shares held']*AAPL_price)
                                +(x['MSFT']*x['shares held']*MSFT_price)
                                +(x['NVDA']*x['shares held']*NVIDIA_price)
                                +(x['F']*x['shares held']*F_price)
                                +(x['GOOGL']*x['shares held']*GOOGL_price),2)
                                ))
print(df)

#db_conn.commit()
drop_table(db_conn, db_cur, 'portfolio_5')
df.to_sql('portfolio_5', engine)
db_conn.commit()
db_conn.close()

