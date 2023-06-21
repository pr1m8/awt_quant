from django.db import models

# Create your models here.
from django.db import models

class StocksPrctarget(models.Model):
    symbol = models.CharField(max_length=255)
    published_date = models.CharField(max_length=255)
    news_url = models.CharField(max_length=255)
    news_title = models.CharField(max_length=255)
    analyst_name = models.CharField(max_length=255)
    price_target = models.FloatField()
    adj_price_target = models.FloatField()
    price_when_posted = models.FloatField()
    news_publisher = models.CharField(max_length=255)
    news_base_url = models.CharField(max_length=255)
    analyst_company = models.CharField(max_length=255)
    symbol = models.CharField(max_length=10,primary_key=True)

    class Meta:
        db_table = 'dim_tickers_stocks_prctarget'
        managed = False
        constraints = [
            models.UniqueConstraint(['symbol', 'date'], name='unique_dim_tickers_stocks_prctarget')
        ]
