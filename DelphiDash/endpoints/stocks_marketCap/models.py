from django.db import models

# Create your models here.
from django.db import models

class StocksMarketCap(models.Model):
    date = models.CharField(max_length=255)
    symbol = models.CharField(max_length=255)
    market_cap = models.IntegerField()
    symbol = models.CharField(max_length=10,primary_key=True)

    class Meta:
        db_table = 'dim_tickers_stocks_market_cap'
        managed = False
        constraints = [
            models.UniqueConstraint(['symbol', 'date'], name='unique_dim_tickers_stocks_market_cap')
        ]
