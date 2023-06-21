from django.db import models

# Create your models here.
from django.db import models

class StocksOffExchange(models.Model):
    symbol = models.CharField(max_length=255)
    date = models.CharField(max_length=255)
    otc_short = models.IntegerField()
    otc_total = models.IntegerField()
    dpi = models.FloatField()
    symbol = models.CharField(max_length=10,primary_key=True)

    class Meta:
        db_table = 'dim_tickers_stocks_off_exchange'
        managed = False
        constraints = [
            models.UniqueConstraint(['symbol', 'date'], name='unique_dim_tickers_stocks_off_exchange')
        ]
