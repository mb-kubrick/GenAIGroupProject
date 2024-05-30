import pandas as pd
import random
import sqlite3
import yfinance as yf
from sqlalchemy import create_engine

def gen_norm_vals() -> list:
    """Gets values between 0 and 1 which some to 1 to get portfolio allocations.
    
    Args:
        None
    """
    values = [round(random.random(), 2) for _ in range(10)]
    total = round(sum(values), 2)
    norm_vals = [round(val/ total, 2) for val in values]
    return norm_vals

def get_insert_list(num_of_clients) -> list:
    """Gets portfolio distributions for every client.
    
    Args:
        num_of_clients: number of clients
    """
    insert_list = []
    for _ in range(num_of_clients):
        values = gen_norm_vals()
        insert_list.append(values)
    print(insert_list)
    return insert_list

def get_shares_list(df) -> list:
    """Gets the number of shares a client owns.
    
    Args:
        df: dataframe of initial client portfolio distribution
    """
    shares_list = []
    for _ in range(len(df)):
        shares_list.append(random.randint(50,100))
    return shares_list

def get_ticker_price(ticker) -> int:
    """Gets the closing stock ticker price from yahoo finance.
    
    Args:
        ticker: stock ticker
    """
    stock = yf.Ticker(ticker)
    data = stock.history(period='1d')
    latest_price = data['Close'].iloc[-1]
    return latest_price

def drop_table(db_conn, db_cur, table_name) -> None:
    """Drops table.
    
    Args:
        db_conn: database connection
        db_cur: database server
        table_name: table name

    Raises:
        Exception: Raised if drop table fails.
    """
    try:
        db_cur.execute(f'''
            DROP TABLE {table_name}
            ''')
        db_conn.commit()
        print(f' table {table_name} dropped')
    except Exception as e:
        print(f'drop_table: {e}')


def create_table(db_conn, db_cur) -> None:
    """Creates table.
    
    Args:
        db_conn: database connection
        db_cur: database server

    Raises:
        Exception: Raised if create table fails.
    """
    try:
        db_cur.execute(f'''
            CREATE TABLE IF NOT EXISTS portfolioTest (
            ClientId INTEGER PRIMARY KEY AUTOINCREMENT,
            AAPL DECIMAL(2,2) NOT NULL,
            MSFT DECIMAL(2,2) NOT NULL,
            NVDA DECIMAL(2,2) NOT NULL,
            F DECIMAL(2,2) NOT NULL,
            GOOGL DECIMAL(2,2) NOT NULL,
            TSLA DECIMAL(2,2) NOT NULL,
            NKE DECIMAL(2,2) NOT NULL,
            META DECIMAL(2,2) NOT NULL,
            AMZN DECIMAL(2,2) NOT NULL,
            ORCL DECIMAL(2,2) NOT NULL
            )
            ''')
        db_conn.commit()
        print(f'table portfolioTest created')
    except Exception as e:
        print(e)

def insert_into_table(db_conn, db_cur, val_list) -> None:
    """Inserts into table.
    
    Args:
        db_conn: database connection
        db_cur: database server
        val_list: client portfolio weightings

    Raises:
        Exception: Raised if insert fails.
    """
    try:
        db_cur.executemany(f'''
            INSERT INTO portfolioTest (
            AAPL, MSFT, NVDA, F, GOOGL, TSLA, NKE, META, AMZN, ORCL)
            VALUES 
            (?,?,?,?,?,?,?,?,?,?)''', val_list)

        db_conn.commit()
        print(f'insert into portfolioTest successful')
    except Exception as e:
        print(e)  




#db_conn = sqlite3.connect('portfolio_allocationsTest.db') 
#db_cur = db_conn.cursor()
#engine = create_engine('sqlite:///portfolio_allocationsTest.db').connect()

def create_synthetic_data(db_conn, db_cur):
    num_of_clients = 15
    insert_list =  get_insert_list(num_of_clients)  

    drop_table(db_conn, db_cur, 'portfolioTest')
    create_table(db_conn, db_cur)
    insert_into_table(db_conn, db_cur, insert_list)
    pass

def get_share_value(db_conn, db_cur, engine):

    AAPL_price = get_ticker_price('AAPL')
    MSFT_price = get_ticker_price('MSFT')
    NVIDIA_price = get_ticker_price('NVDA')
    F_price = get_ticker_price('F')
    GOOGL_price = get_ticker_price('GOOGL')
    TSLA_price = get_ticker_price('TSLA')
    NKE_price = get_ticker_price('NKE')
    META_price = get_ticker_price('META')
    AMZN_price = get_ticker_price('AMZN')
    ORCL_price = get_ticker_price('ORCL')

    df = pd.read_sql_table('portfolioTest', engine)
    shares_list = get_shares_list(df)
    df.insert(11, "shares held", shares_list)

    df=df.assign(value_USD = lambda x: (round((x['AAPL']*x['shares held']*AAPL_price)
                                    +(x['MSFT']*x['shares held']*MSFT_price)
                                    +(x['NVDA']*x['shares held']*NVIDIA_price)
                                    +(x['F']*x['shares held']*F_price)
                                    +(x['GOOGL']*x['shares held']*GOOGL_price)
                                    +(x['TSLA']*x['shares held']*TSLA_price)
                                    +(x['NKE']*x['shares held']*NKE_price)
                                    +(x['META']*x['shares held']*META_price)
                                    +(x['AMZN']*x['shares held']*AMZN_price)
                                    +(x['ORCL']*x['shares held']*ORCL_price),2)
                                    )) 
    

    drop_table(db_conn, db_cur, 'portfolioTest')
    df.to_sql('portfolioTest', engine)
    print(df)
    db_conn.commit()
    db_conn.close()

    pass



