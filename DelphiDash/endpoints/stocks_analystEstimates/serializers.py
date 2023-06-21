from rest_framework import serializers
from .models import *

class StocksAnalystEstimatesSerializer(serializers.ModelSerializer):
    class Meta:
        model = StocksAnalystEstimates
        fields = ('symbol', 'date', 'estimated_revenue_low', 'estimated_revenue_high', 'estimated_revenue_avg', 'estimated_ebitda_low', 'estimated_ebitda_high', 'estimated_ebitda_avg', 'estimated_ebit_low', 'estimated_ebit_high', 'estimated_ebit_avg', 'estimated_net_income_low', 'estimated_net_income_high', 'estimated_net_income_avg', 'estimated_sga_expense_low', 'estimated_sga_expense_high', 'estimated_sga_expense_avg', 'estimated_eps_avg', 'estimated_eps_high', 'estimated_eps_low', 'number_analyst_estimated_revenue', 'number_analysts_estimated_eps',)