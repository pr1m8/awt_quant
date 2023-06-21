from django.db import models

# Create your models here.
from django.db import models

class StocksHstPrcDailyYf(models.Model):
    date = models.DateTimeField()
    open = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    close = models.FloatField()
    volume = models.IntegerField()
    dividends = models.FloatField()
    stock_splits = models.FloatField()
    capital_gains = models.FloatField()
    symbol = models.CharField(max_length=255)
    symbol = models.CharField(max_length=10,primary_key=True)

    class Meta:
        db_table = 'dim_tickers_stocks_hst_prc_daily_yf'
        managed = False
        constraints = [
            models.UniqueConstraint(['symbol', 'date'], name='unique_dim_tickers_stocks_hst_prc_daily_yf')
        ]
