from django.db import models

# Create your models here.
from django.db import models

class StocksPress(models.Model):
    symbol = models.CharField(max_length=255)
    date = models.CharField(max_length=255)
    title = models.CharField(max_length=255)
    text = models.CharField(max_length=255)
    symbol = models.CharField(max_length=10,primary_key=True)

    class Meta:
        db_table = 'dim_tickers_stocks_press'
        managed = False
        constraints = [
            models.UniqueConstraint(['symbol', 'date'], name='unique_dim_tickers_stocks_press')
        ]
