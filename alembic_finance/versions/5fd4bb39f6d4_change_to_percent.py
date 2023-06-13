"""change '%' to 'percent'

Revision ID: 5fd4bb39f6d4
Revises: b3a8cc140bed
Create Date: 2023-06-13 01:40:55.148412

"""
from alembic import op
import sqlalchemy as sa
import re


# revision identifiers, used by Alembic.
revision = '5fd4bb39f6d4'
down_revision = 'b3a8cc140bed'
branch_labels = None
depends_on = None

def camel_to_snake(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

class DatabaseConnection:
    def __init__(self, database):
        self.fin_engine = sa.create_engine(f"postgresql://postgres:postgres@192.168.2.69:5432/{database}")
                                                   

    def get_column_names(self, table_name):
        with self.fin_engine.connect() as connection:
            query = f"SELECT column_name FROM information_schema.columns WHERE table_name='{table_name}';"
            result = connection.execute(sa.text(query))
            return [row[0] for row in result]
        
    def get_table_names(self):
        with self.fin_engine.connect() as connection:
            query = f"SELECT table_name FROM information_schema.tables WHERE table_schema='public';"
            result = connection.execute(sa.text(query))
            return [row[0] for row in result]
        
    def update_column_names(self, table_name):
        with self.fin_engine.connect() as connection:
            for column_name in self.get_column_names(table_name):
                mod_column_name = camel_to_snake(column_name)
                if column_name == mod_column_name:
                    continue
                query = f'ALTER TABLE {table_name} RENAME "{column_name}" TO {mod_column_name};'
                connection.execute(sa.text(query))
        
conn = DatabaseConnection("finance")
tables = conn.get_table_names()

cleaned_tables = []
for table in tables:
    if  table.startswith("dim") or table.startswith("fct"):
        cleaned_tables.append(table)

def upgrade() -> None:
    for table in cleaned_tables:
        for column in conn.get_column_names(table):
            new_column_name = column.replace("%", "pct")
            if column == new_column_name:
                continue
            op.alter_column(table, column, nullable=True, new_column_name=new_column_name)

def downgrade() -> None:
    pass
