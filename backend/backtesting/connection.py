import psycopg2
from config import load_config

class DatabaseConnection:
    def __init__(self):
        self.conn=None

    def connect(self):
        try:
            params = load_config()
            self.conn = psycopg2.connect(**params)
            print("Connection successful")
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

    def create_table(self,query):
        cursor=self.conn.cursor()
        cursor.execute(query)
        self.conn.commit()
        cursor.close()
        self.conn.close()

db=DatabaseConnection()
drop_tables_query = """
DO $$ DECLARE
    r RECORD;
BEGIN
    -- For each table in the public schema
    FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = 'public') LOOP
        EXECUTE 'DROP TABLE IF EXISTS ' || quote_ident(r.tablename) || ' CASCADE';
    END LOOP;
END $$;
"""


table1="""
CREATE TABLE IF NOT EXISTS forecasting_algorithms(
    alg_id SERIAL PRIMARY KEY,
    name varchar(26)
);
"""
table2="""
CREATE TABLE IF NOT EXISTS real_time_forecast_datetime (
    date_id SERIAL PRIMARY KEY,
    Datetime TIMESTAMP WITHOUT TIME ZONE,
    news_description varchar(26) NULL,
    news_source varchar(26) NULL
);
"""
table3="""
CREATE TABLE IF NOT EXISTS backtest_simulation (
    simulation_id SERIAL PRIMARY KEY,
    training_period varchar(26)
);

"""
table4="""
CREATE TABLE IF NOT EXISTS stock_data (
    stock_id SERIAL PRIMARY KEY,
    stock_name varchar(26),
    stock_ticker varchar(26)
);
"""


table5="""
CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id SERIAL PRIMARY KEY,
    forecast_value REAL not null,
    date_id INTEGER NOT NULL,
    accuracy INTEGER NULL,
    simulation_id integer not null,
    timestamp TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    alg_id INTEGER NOT NULL,
    stock_id integer not null,
    FOREIGN KEY (alg_id) REFERENCES forecasting_algorithms(alg_id),
    FOREIGN KEY (date_id) REFERENCES real_time_forecast_datetime(date_id),
    FOREIGN KEY (stock_id) REFERENCES stock_data(stock_id),
    FOREIGN KEY (simulation_id) REFERENCES backtest_simulation(simulation_id)
);
"""


for i in [drop_tables_query,table1,table2,table3,table4,table5]:
    db.connect()
    db.create_table(i)

