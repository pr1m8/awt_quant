from django.db import models

# Create your models here.
from django.db import models

class StocksPrcDailyAll(models.Model):
    date = models.CharField(max_length=255)
    symbol = models.CharField(max_length=255)
    open = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    close = models.FloatField()
    adj_close = models.FloatField()
    volume = models.IntegerField()
    symbol = models.CharField(max_length=10,primary_key=True)

    class Meta:
        db_table = 'dim_tickers_stocks_prc_daily_all'
        managed = False
        constraints = [
            models.UniqueConstraint(['symbol', 'date'], name='unique_dim_tickers_stocks_prc_daily_all')
        ]
