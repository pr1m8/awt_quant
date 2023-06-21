from django.db import models

# Create your models here.
from django.db import models

class StocksCorporatelobbyingQvr(models.Model):
    symbol = models.CharField(max_length=255)
    date = models.CharField(max_length=255)
    client = models.CharField(max_length=255)
    amount = models.FloatField()
    issue = models.CharField(max_length=255)
    specific_issue = models.CharField(max_length=255)
    symbol = models.CharField(max_length=10,primary_key=True)

    class Meta:
        db_table = 'dim_tickers_stocks_corporatelobbying_qvr'
        managed = False
        constraints = [
            models.UniqueConstraint(['symbol', 'date'], name='unique_dim_tickers_stocks_corporatelobbying_qvr')
        ]
