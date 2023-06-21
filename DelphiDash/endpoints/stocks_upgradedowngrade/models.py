from django.db import models

# Create your models here.
from django.db import models

class StocksUpgradedowngrade(models.Model):
    symbol = models.CharField(max_length=255)
    published_date = models.CharField(max_length=255)
    news_url = models.CharField(max_length=255)
    news_title = models.CharField(max_length=255)
    news_base_url = models.CharField(max_length=255)
    news_publisher = models.CharField(max_length=255)
    new_grade = models.CharField(max_length=255)
    previous_grade = models.CharField(max_length=255)
    grading_company = models.CharField(max_length=255)
    action = models.CharField(max_length=255)
    price_when_posted = models.FloatField()
    symbol = models.CharField(max_length=10,primary_key=True)

    class Meta:
        db_table = 'dim_tickers_stocks_upgradedowngrade'
        managed = False
        constraints = [
            models.UniqueConstraint(['symbol', 'date'], name='unique_dim_tickers_stocks_upgradedowngrade')
        ]
