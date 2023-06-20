from rest_framework import serializers
from .models import StocksHstEmployees

class StocksHstEmployeesSerializer(serializers.ModelSerializer):
    class Meta:
        model = StocksHstEmployees
        fields = ('symbol', 'filing_date', 'employee_count', 'cik', 'acceptance_time', 'period_of_report', 'company_name', 'form_type')
                  
