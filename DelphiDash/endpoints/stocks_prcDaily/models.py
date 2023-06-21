from django.db import models

class StocksPrcDaily(models.Model):    
    date = models.CharField(max_length=255)
    open = models.FloatField()
    close = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    adj_close = models.FloatField()
    volume = models.IntegerField()
    symbol = models.CharField(max_length=10,primary_key=True)

    class Meta:
        db_table = 'dim_tickers_stocks_prc_daily'
        managed = False
        constraints = [
            models.UniqueConstraint(['symbol', 'date'], name='unique_dim_tickers_stocks_prc_daily')
        ]
