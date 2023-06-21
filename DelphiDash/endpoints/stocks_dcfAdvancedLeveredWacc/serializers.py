from rest_framework import serializers
from .models import *

class StocksDcfAdvancedLeveredWaccSerializer(serializers.ModelSerializer):
    class Meta:
        model = StocksDcfAdvancedLeveredWacc
        fields = ('year', 'symbol', 'revenue', 'revenue_percentage', 'capital_expenditure', 'capital_expenditure_percentage', 'price', 'beta', 'diluted_shares_outstanding', 'cost_of_debt', 'tax_rate', 'after_tax_cost_of_debt', 'risk_free_rate', 'market_risk_premium', 'cost_of_equity', 'total_debt', 'total_equity', 'total_capital', 'debt_weighting', 'equity_weighting', 'wacc', 'operating_cash_flow', 'pv_lfcf', 'sum_pv_lfcf', 'long_term_growth_rate', 'free_cash_flow', 'terminal_value', 'present_terminal_value', 'enterprise_value', 'net_debt', 'equity_value', 'equity_value_per_share', 'free_cash_flow_t1', 'operating_cash_flow_percentage',)