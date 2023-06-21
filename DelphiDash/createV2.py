import subprocess
import os
import sqlalchemy

class Djangautomate:
    def __init__(self, db_engine, table_name, index_cols=None, app_name='', project_name=''):
        self.db_engine = db_engine
        self.table_name = table_name
        self.index_cols = index_cols
        self.app_name = app_name
        self.project_name = project_name

        self.camelcased_app_name = "".join([part[0].upper()+part[1:] for part in  self.app_name.split('_')])
        self.viewset_name = f"{self.camelcased_app_name}ViewSet"
        self.serializer_name = f"{self.camelcased_app_name}Serializer"

    def create_django_app(self):
        subprocess.run(['django-admin', 'startapp', self.app_name])

     
    def get_model_field_type(self, col_type):
        if isinstance(col_type, sqlalchemy.String):
            return "CharField(max_length=255)"
        elif isinstance(col_type, sqlalchemy.Integer):
            return "IntegerField()"
        elif isinstance(col_type, sqlalchemy.Float):
            return "FloatField()"
        elif isinstance(col_type, sqlalchemy.Boolean):
            return "BooleanField()"
        elif isinstance(col_type, sqlalchemy.DateTime):
            return "DateTimeField()"
        else:
            return None

    def generate_model_code(self):
        model_code =  f"from django.db import models\n\n"
        model_code += f"class {self.camelcased_app_name}(models.Model):\n"
        md = sqlalchemy.MetaData()
        table = sqlalchemy.Table(self.table_name, md, autoload_with=self.db_engine)
        # Generate model fields based on column data types
        columns = table.c
        for col in columns:
            if col.name=='index':
                continue
            col_name = col.name
            col_type = col.type

            field_type = self.get_model_field_type(col_type)

            if field_type:
                model_code += f"    {col_name} = models.{field_type}\n"

        # Add id field if symbol and date are not present
        if "symbol" not in columns and "date" not in columns:
            model_code += "    id = models.AutoField(primary_key=True)\n"
        elif 'symbol' in columns:
            model_code+="    symbol = models.CharField(max_length=10,primary_key=True)\n"
        model_code += f"\n    class Meta:\n"
        model_code += f"        db_table = '{self.table_name}'\n"
        model_code += f"        managed = False\n"
        model_code += f"        constraints = [\n"
        model_code += f"            models.UniqueConstraint({self.index_cols}, name='unique_{self.table_name}')\n"
        model_code += f"        ]\n"

        return model_code
        
    def generate_view_code(self):
        view_code =  f"from .models import *\n"
        view_code += f"from .serializers import *\n"
        view_code += f"from rest_framework import viewsets\n"
        view_code += f"from rest_framework.response import Response\n\n"
        view_code += f"class {self.viewset_name}(viewsets.ModelViewSet):\n"
        view_code += f"    serializer_class = {self.serializer_name}\n"
        view_code += f"    queryset = {self.camelcased_app_name}.objects.all()\n\n"
        view_code += f"    def list(self, request):\n"
        view_code += f"        queryset = {self.camelcased_app_name}.objects.all()\n"
        view_code += f"        serializer = {self.serializer_name}(queryset, many=True)\n"
        view_code += f"        return Response(serializer.data)\n\n"
        view_code += f"    def retrieve(self, request, pk=None):\n"
        view_code += f"        queryset = {self.camelcased_app_name}.objects.filter(symbol=pk)\n"
        view_code += f"        serializer = {self.serializer_name}(queryset, many=True)\n"
        view_code += f"        return Response(serializer.data)"

        return view_code
    
    def generate_serializer_code(self):
        serializer_code =  f"from rest_framework import serializers\n"
        serializer_code += f"from .models import *\n\n"
        serializer_code += f"class {self.serializer_name}(serializers.ModelSerializer):\n"
        serializer_code += f"    class Meta:\n"
        serializer_code += f"        model = {self.camelcased_app_name}\n"
        serializer_code += f"        fields = ("

        md = sqlalchemy.MetaData()
        table = sqlalchemy.Table(self.table_name, md, autoload_with=self.db_engine)
        columns = table.c
        for col in columns:
            if col.name=='index':
                continue
            serializer_code += f"'{col.name}', "
        
        serializer_code = serializer_code[:-1] + ")"
        return serializer_code
            
    def generate_app_code(self):
        app_code =  f"from django.apps import AppConfig\n\n"
        app_code += f"class {self.camelcased_app_name}Config(AppConfig):\n"
        app_code += f"    default_auto_field = 'django.db.models.BigAutoField'\n"
        app_code += f"    name = 'endpoints.{self.app_name}'"

        return app_code

    def generate_template_code(self):
        template_code = "<!-- Write your HTML template code here -->\n"
        return template_code


    def generate_code_files(self):
        # Check if the app directory exists, otherwise create the app
        app_directory = os.path.join(os.getcwd(), self.app_name)
        if not os.path.exists(app_directory):
            self.create_django_app()

        model_code = self.generate_model_code()
        view_code = self.generate_view_code()
        template_code = self.generate_template_code()
        serializer_code = self.generate_serializer_code()
        app_code = self.generate_app_code()

        model_file_path = os.path.join(app_directory, "models.py")
        view_file_path = os.path.join(app_directory, "views.py")
        template_file_path = os.path.join(app_directory, "templates", self.app_name, self.project_name + ".html")
        serializer_file_path = os.path.join(app_directory, "serializers.py")
        app_file_path = os.path.join(app_directory, "apps.py")

        urls_file_path = os.path.join(os.getcwd(), '..', r'delphi_api', "urls.py")
        settings_file_path = os.path.join(os.getcwd(), '..', r'delphi_api', "settings.py")
        static_directory = os.path.join(app_directory, "static")
        css_directory = os.path.join(static_directory, self.app_name, "css")
        js_directory = os.path.join(static_directory, self.app_name, "js")

        # Append to existing files if they exist
        mode = 'a' if os.path.isfile(model_file_path) else 'w'
        with open(model_file_path, mode) as model_file:
            model_file.write(model_code)

        mode = 'a' if os.path.isfile(view_file_path) else 'w'
        with open(view_file_path, mode) as view_file:
            view_file.write(view_code)

        mode = 'a' if os.path.isfile(serializer_file_path) else 'w'
        with open(serializer_file_path, mode) as serializer_file:
            serializer_file.write(serializer_code)

        with open(app_file_path, "w") as app_file:
            app_file.write(app_code)

        # Append to existing URL file or create a new one
        existing_urls = []
        with open(urls_file_path, 'r') as url_file:
            existing_urls = url_file.readlines()

        # Update the URL patterns
        updated_urls = []
        updated_urls.append(f"from endpoints.{self.app_name}.views import {self.viewset_name}\n")
        for line in existing_urls:
            if line.startswith("router = "):
                updated_urls.append(line)
                updated_urls.append(
                    f"router.register(r'{self.camelcased_app_name}', {self.viewset_name}, '{self.camelcased_app_name}')\n")
            else:
                updated_urls.append(line)

        # Write the updated URL file
        with open(urls_file_path, 'w') as url_file:
            url_file.writelines(updated_urls)

        mode = 'a' if os.path.isfile(template_file_path) else 'w'
        os.makedirs(os.path.dirname(template_file_path), exist_ok=True)
        with open(template_file_path, mode) as template_file:
            template_file.write(template_code)

        os.makedirs(css_directory, exist_ok=True)
        os.makedirs(js_directory, exist_ok=True)

        print(f"Code files generated for {self.project_name} successfully!")

        # Update settings.py
        with open(settings_file_path, 'a') as settings_file:
            settings_file.write(f"INSTALLED_APPS.append('endpoints.{self.app_name}.apps.{self.camelcased_app_name}Config')\n")

        print(f"{self.app_name} added to INSTALLED_APPS in settings.py!")
        print(f"{self.app_name} URLs added to {self.project_name}/urls.py successfully!")



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

automated_tables = []
for table in tables:
    if table == "dim_tickers_stocks_prc_daily":
        continue
    if "dim_tickers_stocks" in table:
        automated_tables.append(table)



fin_engine = conn.fin_engine

for table in automated_tables:
    labels = table.split("_")
    app_name= "stocks_" + labels[3]
    for label in labels[4:]:
        app_name += label.capitalize()

    generator = Djangautomate(fin_engine,table,app_name=app_name,index_cols=["symbol", "date"])
    generator.generate_code_files()