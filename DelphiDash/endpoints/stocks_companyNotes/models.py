from django.db import models

# Create your models here.
from django.db import models

class StocksCompanyNotes(models.Model):
    symbol = models.CharField(max_length=255)
    cik = models.IntegerField()
    title = models.CharField(max_length=255)
    exchange = models.CharField(max_length=255)
    symbol = models.CharField(max_length=10,primary_key=True)

    class Meta:
        db_table = 'dim_tickers_stocks_companyNotes'
        managed = False
        constraints = [
            models.UniqueConstraint(['symbol', 'date'], name='unique_dim_tickers_stocks_companyNotes')
        ]
