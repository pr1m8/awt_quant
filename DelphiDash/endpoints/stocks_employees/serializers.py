from rest_framework import serializers
from .models import *

class StocksEmployeesSerializer(serializers.ModelSerializer):
    class Meta:
        model = StocksEmployees
        fields = ('symbol', 'cik', 'filing_date', 'acceptance_time', 'period_of_report', 'company_name', 'form_type', 'employee_count', 'source',)