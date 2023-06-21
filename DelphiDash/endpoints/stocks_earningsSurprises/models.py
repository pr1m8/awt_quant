from django.db import models

# Create your models here.
from django.db import models

class StocksEarningsSurprises(models.Model):
    date = models.CharField(max_length=255)
    symbol = models.CharField(max_length=255)
    actual_earning_result = models.FloatField()
    estimated_earning = models.FloatField()
    symbol = models.CharField(max_length=10,primary_key=True)

    class Meta:
        db_table = 'dim_tickers_stocks_earnings_surprises'
        managed = False
        constraints = [
            models.UniqueConstraint(['symbol', 'date'], name='unique_dim_tickers_stocks_earnings_surprises')
        ]
