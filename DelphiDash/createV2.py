import subprocess
import os

import pandas as pd
import sqlalchemy


import subprocess
import os
import sqlalchemy
import tqdm
import subprocess
import os
import sqlalchemy
import tqdm
#NEED TO WRITE CONFIG INTO SETTINGS

class Djangautomate:
    def __init__(self, db_engine, table_names, index_cols=None, app_name='', project_name=''):
        self.db_engine = db_engine
        self.table_names = table_names
        self.index_cols = index_cols
        self.app_name = app_name
        self.project_name = project_name

        camelcased_app_name = self.app_name.replace("_", "").capitalize()
        view_function_name = f"{camelcased_app_name}View"
        self.vw_func_name = view_function_name

    def create_django_app(self):
        subprocess.run(['django-admin', 'startapp', self.app_name])

    def generate_model_code(self,):
        table_name=self.table_names[0]

        model_code = f"class {self.vw_func_name.replace('View','')}(models.Model):\n"
        md = sqlalchemy.MetaData()
        table = sqlalchemy.Table(table_name, md, autoload_with=self.db_engine)
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
        model_code += f"        db_table = '{table_name}'\n"
        model_code += f"        managed = False\n"
        model_code += f"        constraints = [\n"
        model_code += f"            models.UniqueConstraint({self.index_cols}, name='unique_{table_name}')\n"
        model_code += f"        ]\n"

        return model_code
    
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
        
    def generate_view_code(self, ):
        table_name = self.table_names[0]


        #model_code = f"class {table_name.capitalize()}(models.Model):\n"
        md = sqlalchemy.MetaData()
        table = sqlalchemy.Table(self.table_names[0], md, autoload_with=self.db_engine)
        camelcased_app_name = self.app_name.replace("_", "").capitalize()
        view_function_name = f"{camelcased_app_name}View"
        self.vw_func_name = view_function_name
        columns = table.c
        view_code = f"from rest_framework.views import APIView\n"
        view_code += f"from .models import *\n\n"
        view_code += f"from rest_framework.response import Response\n\n"
        view_code += f"class {view_function_name}(APIView):\n"
        view_code += f"    def get(self, request, symbol):\n"
        view_code += f"        stock_p = {camelcased_app_name}.objects.filter(symbol=symbol).using('finance')\n"
        view_code += f"        db_table= '{table_name}'\n"
        view_code += f"        data = []\n"
        view_code += f"        for item in stock_p:\n"
        view_code += f"            data.append({{"
        #table = self.db_engine.table(table_name)
        columns = table.columns
        # Generate view response data fields based on columns
        for col in columns:
            if col.name=='index':
                continue
            col_name = col.name
            view_code += f"'{col_name}': item.{col_name}, "

        view_code += f"}})\n"
        view_code += f"        return Response(data)\n"

        return view_code

    def generate_template_code(self):
        template_code = "<!-- Write your HTML template code here -->\n"
        return template_code

    def generate_url_code(self):
        url_code = f"path('api/{self.app_name}/{self.project_name}/<str:symbol>/', views.{self.project_name.capitalize()}View.as_view(), name='{self.app_name}-{self.project_name}')\n"
        return url_code

    def generate_code_files(self):
        # Check if the app directory exists, otherwise create the app
        app_directory = os.path.join(os.getcwd(), self.app_name)
        if not os.path.exists(app_directory):
            self.create_django_app()

        model_code = self.generate_model_code()
        view_code = self.generate_view_code()
        template_code = self.generate_template_code()
        url_code = self.generate_url_code()

        model_file_path = os.path.join(app_directory, "models.py")
        view_file_path = os.path.join(app_directory, "views.py")
        template_file_path = os.path.join(app_directory, "templates", self.app_name, self.project_name + ".html")

        urls_file_path = os.path.join(os.getcwd(), r'delphi_api', "urls.py")
        print(urls_file_path)
        static_directory = os.path.join(app_directory, "static")
        css_directory = os.path.join(static_directory, self.app_name, "css")
        js_directory = os.path.join(static_directory, self.app_name, "js")

        # Append to existing files if they exist
        mode = 'a' if os.path.isfile(model_file_path) else 'w'
        with open(model_file_path, mode) as model_file:
            if mode == 'w':
                model_file.write(f"from django.db import models\n\n")
            model_file.write(model_code)

        mode = 'a' if os.path.isfile(view_file_path) else 'w'
        # Camelcase the app name
        camelcased_app_name = ''.join(word.capitalize() for word in self.app_name.split('_'))

        # Create the camelcased function name
        function_name = camelcased_app_name[0].lower() + camelcased_app_name[1:] + "View"
        with open(view_file_path, mode) as view_file:
            if mode == 'w':
                view_file.write(f"from rest_framework.views import APIView\n")
                view_file.write(f"from rest_framework.response import Response\n\n")
                view_file.write(f"from .models import {camelcased_app_name}\n\n")
            view_file.write(view_code)

        # Append to existing URL file or create a new one
        existing_urls = []
        if os.path.exists(urls_file_path):
            with open(urls_file_path, 'r') as url_file:
                existing_urls = url_file.readlines()

        # Update the URL patterns
        updated_urls = []
        updated_urls.append(f"from {self.app_name}.views import {self.vw_func_name}\n")
        for line in existing_urls:
            if line.startswith("from "):
                updated_urls.append(line)
            elif line.startswith("urlpatterns = ["):
                updated_urls.append(line)
                updated_urls.append(
                    f"    path('api/{self.app_name}/<str:symbol>/', {self.vw_func_name}.as_view(), name='{self.app_name}'),\n")
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
        settings_file_path = os.path.join(r"delphi_api", "settings.py")
        with open(settings_file_path, 'a') as settings_file:
            settings_file.write(f"\n# Added by Djangautomate\n")
            settings_file.write(f"INSTALLED_APPS.append('{self.app_name}.apps.{camelcased_app_name}Config')\n")

        print(f"{self.app_name} added to INSTALLED_APPS in settings.py!")
        print(f"{self.app_name} URLs added to {self.project_name}/urls.py successfully!")

fin_engine = sqlalchemy.create_engine(str("postgresql://postgres:postgres@192.168.2.69:5432/finance"))
app_name='stocks_hstEmployees'
table_names = ['dim_ticker_stocks_hst_employees_fmp',]#'dim_tickers_stocks_market_cap']
generator = Djangautomate(fin_engine,table_names,app_name=app_name,index_cols=["symbol", "date"])
generator.generate_code_files()