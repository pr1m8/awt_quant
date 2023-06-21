from django.db import models

# Create your models here.
from django.db import models

class StocksKeyExecutives(models.Model):
    title = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    pay = models.CharField(max_length=255)
    currency_pay = models.CharField(max_length=255)
    gender = models.CharField(max_length=255)
    year_born = models.CharField(max_length=255)
    title_since = models.FloatField()
    id = models.AutoField(primary_key=True)

    class Meta:
        db_table = 'dim_tickers_stocks_key_executives'
        managed = False
        constraints = [
            models.UniqueConstraint(['symbol', 'date'], name='unique_dim_tickers_stocks_key_executives')
        ]
