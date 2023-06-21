from django.db import models

# Create your models here.
from django.db import models

class StocksAnalystEstimates(models.Model):
    symbol = models.CharField(max_length=255)
    date = models.CharField(max_length=255)
    estimated_revenue_low = models.IntegerField()
    estimated_revenue_high = models.IntegerField()
    estimated_revenue_avg = models.IntegerField()
    estimated_ebitda_low = models.FloatField()
    estimated_ebitda_high = models.FloatField()
    estimated_ebitda_avg = models.FloatField()
    estimated_ebit_low = models.FloatField()
    estimated_ebit_high = models.FloatField()
    estimated_ebit_avg = models.FloatField()
    estimated_net_income_low = models.FloatField()
    estimated_net_income_high = models.FloatField()
    estimated_net_income_avg = models.FloatField()
    estimated_sga_expense_low = models.FloatField()
    estimated_sga_expense_high = models.FloatField()
    estimated_sga_expense_avg = models.FloatField()
    estimated_eps_avg = models.FloatField()
    estimated_eps_high = models.FloatField()
    estimated_eps_low = models.FloatField()
    number_analyst_estimated_revenue = models.IntegerField()
    number_analysts_estimated_eps = models.IntegerField()
    symbol = models.CharField(max_length=10,primary_key=True)

    class Meta:
        db_table = 'dim_tickers_stocks_analyst_estimates'
        managed = False
        constraints = [
            models.UniqueConstraint(['symbol', 'date'], name='unique_dim_tickers_stocks_analyst_estimates')
        ]
