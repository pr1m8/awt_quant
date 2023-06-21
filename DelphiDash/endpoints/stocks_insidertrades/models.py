from django.db import models

# Create your models here.
from django.db import models

class StocksInsidertrades(models.Model):
    symbol = models.CharField(max_length=255)
    filing_date = models.CharField(max_length=255)
    transaction_date = models.CharField(max_length=255)
    reporting_cik = models.CharField(max_length=255)
    transaction_type = models.CharField(max_length=255)
    securities_owned = models.CharField(max_length=255)
    company_cik = models.CharField(max_length=255)
    reporting_name = models.CharField(max_length=255)
    type_of_owner = models.CharField(max_length=255)
    acquistion_or_disposition = models.CharField(max_length=255)
    form_type = models.CharField(max_length=255)
    securities_transacted = models.CharField(max_length=255)
    price = models.CharField(max_length=255)
    security_name = models.CharField(max_length=255)
    link = models.CharField(max_length=255)
    symbol = models.CharField(max_length=10,primary_key=True)

    class Meta:
        db_table = 'dim_tickers_stocks_insidertrades'
        managed = False
        constraints = [
            models.UniqueConstraint(['symbol', 'date'], name='unique_dim_tickers_stocks_insidertrades')
        ]
