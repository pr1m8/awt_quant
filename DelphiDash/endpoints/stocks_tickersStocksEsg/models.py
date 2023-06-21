from django.db import models

# Create your models here.
from django.db import models

class StocksTickersStocksEsg(models.Model):
    group_type = models.CharField(max_length=255)
    group_index = models.CharField(max_length=255)
    date = models.CharField(max_length=255)
    environmentalScore_mean = models.FloatField()
    environmentalScore_std = models.FloatField()
    environmentalScore_var = models.FloatField()
    environmentalScore_median = models.FloatField()
    environmentalScore_q25 = models.FloatField()
    environmentalScore_q75 = models.FloatField()
    environmentalScore_min = models.FloatField()
    environmentalScore_max = models.FloatField()
    environmentalScore_count = models.IntegerField()
    socialScore_mean = models.FloatField()
    socialScore_std = models.FloatField()
    socialScore_var = models.FloatField()
    socialScore_median = models.FloatField()
    socialScore_q25 = models.FloatField()
    socialScore_q75 = models.FloatField()
    socialScore_min = models.FloatField()
    socialScore_max = models.FloatField()
    socialScore_count = models.IntegerField()
    governanceScore_mean = models.FloatField()
    governanceScore_std = models.FloatField()
    governanceScore_var = models.FloatField()
    governanceScore_median = models.FloatField()
    governanceScore_q25 = models.FloatField()
    governanceScore_q75 = models.FloatField()
    governanceScore_min = models.FloatField()
    governanceScore_max = models.FloatField()
    governanceScore_count = models.IntegerField()
    ESGScore_mean = models.FloatField()
    ESGScore_std = models.FloatField()
    ESGScore_var = models.FloatField()
    ESGScore_median = models.FloatField()
    ESGScore_q25 = models.FloatField()
    ESGScore_q75 = models.FloatField()
    ESGScore_min = models.FloatField()
    ESGScore_max = models.FloatField()
    ESGScore_count = models.IntegerField()

    class Meta:
        db_table = 'vw_cat_dim_tickers_stocks_esg'
        managed = False
        constraints = [
            models.UniqueConstraint(['symbol', 'date'], name='unique_vw_cat_dim_tickers_stocks_esg')
        ]
