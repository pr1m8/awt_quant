from django.db import models

# Create your models here.
from django.db import models

class StocksCorporatelobbying(models.Model):
    date = models.CharField(max_length=255)
    symbol = models.CharField(max_length=255)
    client = models.CharField(max_length=255)
    amount = models.FloatField()
    issue = models.CharField(max_length=255)
    specific_issue = models.CharField(max_length=255)
    registrant = models.CharField(max_length=255)
    symbol = models.CharField(max_length=10,primary_key=True)

    class Meta:
        db_table = 'dim_tickers_stocks_corporatelobbying'
        managed = False
        constraints = [
            models.UniqueConstraint(['symbol', 'date'], name='unique_dim_tickers_stocks_corporatelobbying')
        ]
