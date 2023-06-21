from django.db import models

# Create your models here.
from django.db import models

class StocksDcfDaily(models.Model):
    date = models.CharField(max_length=255)
    symbol = models.CharField(max_length=255)
    dcf = models.FloatField()
    symbol = models.CharField(max_length=10,primary_key=True)

    class Meta:
        db_table = 'dim_tickers_stocks_dcf_daily'
        managed = False
        constraints = [
            models.UniqueConstraint(['symbol', 'date'], name='unique_dim_tickers_stocks_dcf_daily')
        ]
