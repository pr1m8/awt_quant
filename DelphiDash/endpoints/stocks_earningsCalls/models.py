from django.db import models

# Create your models here.
from django.db import models

class StocksEarningsCalls(models.Model):
    date = models.CharField(max_length=255)
    symbol = models.CharField(max_length=255)
    quarter = models.IntegerField()
    year = models.IntegerField()
    content = models.CharField(max_length=255)
    symbol = models.CharField(max_length=10,primary_key=True)

    class Meta:
        db_table = 'dim_tickers_stocks_earnings_calls'
        managed = False
        constraints = [
            models.UniqueConstraint(['symbol', 'date'], name='unique_dim_tickers_stocks_earnings_calls')
        ]
