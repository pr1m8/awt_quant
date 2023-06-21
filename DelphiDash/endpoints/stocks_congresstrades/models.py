from django.db import models

# Create your models here.
from django.db import models

class StocksCongresstrades(models.Model):
    report_date = models.CharField(max_length=255)
    transaction_date = models.CharField(max_length=255)
    symbol = models.CharField(max_length=255)
    representative = models.CharField(max_length=255)
    transaction = models.CharField(max_length=255)
    amount = models.FloatField()
    house = models.CharField(max_length=255)
    range = models.CharField(max_length=255)
    symbol = models.CharField(max_length=10,primary_key=True)

    class Meta:
        db_table = 'dim_tickers_stocks_congresstrades'
        managed = False
        constraints = [
            models.UniqueConstraint(['symbol', 'date'], name='unique_dim_tickers_stocks_congresstrades')
        ]
