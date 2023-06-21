from django.db import models

# Create your models here.
from django.db import models

class StocksDcfAdvancedLeveredWacc(models.Model):
    year = models.IntegerField()
    symbol = models.CharField(max_length=255)
    revenue = models.FloatField()
    revenue_percentage = models.FloatField()
    capital_expenditure = models.FloatField()
    capital_expenditure_percentage = models.FloatField()
    price = models.FloatField()
    beta = models.FloatField()
    diluted_shares_outstanding = models.FloatField()
    cost_of_debt = models.FloatField()
    tax_rate = models.FloatField()
    after_tax_cost_of_debt = models.FloatField()
    risk_free_rate = models.FloatField()
    market_risk_premium = models.FloatField()
    cost_of_equity = models.FloatField()
    total_debt = models.FloatField()
    total_equity = models.FloatField()
    total_capital = models.FloatField()
    debt_weighting = models.FloatField()
    equity_weighting = models.FloatField()
    wacc = models.FloatField()
    operating_cash_flow = models.FloatField()
    pv_lfcf = models.FloatField()
    sum_pv_lfcf = models.FloatField()
    long_term_growth_rate = models.FloatField()
    free_cash_flow = models.FloatField()
    terminal_value = models.CharField(max_length=255)
    present_terminal_value = models.FloatField()
    enterprise_value = models.CharField(max_length=255)
    net_debt = models.FloatField()
    equity_value = models.CharField(max_length=255)
    equity_value_per_share = models.FloatField()
    free_cash_flow_t1 = models.FloatField()
    operating_cash_flow_percentage = models.FloatField()
    symbol = models.CharField(max_length=10,primary_key=True)

    class Meta:
        db_table = 'dim_tickers_stocks_dcf_advanced_levered_wacc'
        managed = False
        constraints = [
            models.UniqueConstraint(['symbol', 'date'], name='unique_dim_tickers_stocks_dcf_advanced_levered_wacc')
        ]
