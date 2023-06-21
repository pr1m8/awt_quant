from django.db import models

# Create your models here.
from django.db import models

class StocksEsgrisk(models.Model):
    symbol = models.CharField(max_length=255)
    date = models.CharField(max_length=255)
    cik = models.CharField(max_length=255)
    company_name = models.CharField(max_length=255)
    form_type = models.CharField(max_length=255)
    accepted_date = models.CharField(max_length=255)
    environmental_score = models.FloatField()
    social_score = models.FloatField()
    governance_score = models.FloatField()
    esg_score = models.FloatField()
    url = models.CharField(max_length=255)
    symbol = models.CharField(max_length=10,primary_key=True)

    class Meta:
        db_table = 'dim_tickers_stocks_esgrisk'
        managed = False
        constraints = [
            models.UniqueConstraint(['symbol', 'date'], name='unique_dim_tickers_stocks_esgrisk')
        ]
