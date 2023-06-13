"""
MADE BY Thomas Mayer

This will be built upon for internal dev use to access the database.
"""

import sqlalchemy
import re
import alembic

def camel_to_snake(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

class DatabaseConnection:
    def __init__(self, database):
        self.fin_engine = sqlalchemy.create_engine(f"postgresql://postgres:postgres@192.168.2.69:5432/{database}")
                                                   

    def get_column_names(self, table_name):
        with self.fin_engine.connect() as connection:
            query = f"SELECT column_name FROM information_schema.columns WHERE table_name='{table_name}';"
            result = connection.execute(sqlalchemy.text(query))
            return [row[0] for row in result]
        
    def get_table_names(self):
        with self.fin_engine.connect() as connection:
            query = f"SELECT table_name FROM information_schema.tables WHERE table_schema='public';"
            result = connection.execute(sqlalchemy.text(query))
            return [row[0] for row in result]
        
    def update_column_names(self, table_name):
        with self.fin_engine.connect() as connection:
            for column_name in self.get_column_names(table_name):
                mod_column_name = camel_to_snake(column_name)
                if column_name == mod_column_name:
                    continue
                query = f'ALTER TABLE {table_name} RENAME "{column_name}" TO {mod_column_name};'
                connection.execute(sqlalchemy.text(query))
        
conn = DatabaseConnection("finance")
tables = conn.get_table_names()

cleaned_tables = []
for table in tables:
    if  table.startswith("dim") or table.startswith("fct"):
        cleaned_tables.append(table)

print(cleaned_tables[0])

# for table in cleaned_tables:
#     print(conn.get_column_names(table))

column_names = conn.get_column_names("dim_countries_market_risk_premium")
print(column_names)
conn.update_column_names("dim_countries_market_risk_premium")

# alembic.alter_column("dim_countries_market_risk_premium", "countryRiskPremium", nullable=False, new_column_name="country_risk_")

