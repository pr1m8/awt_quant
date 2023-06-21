from django.db import models

# Create your models here.
from django.db import models

class StocksPrcQntGbmFcst(models.Model):
    mean_path = models.FloatField()
    median_path = models.FloatField()
    path_99 = models.FloatField()
    path_75 = models.FloatField()
    path_25 = models.FloatField()
    path_01 = models.FloatField()
    min_path = models.FloatField()
    max_path = models.FloatField()
    stock = models.CharField(max_length=255)
    total_days = models.IntegerField()
    training_days = models.IntegerField()
    forecast_days = models.IntegerField()
    bounded = models.BooleanField()
    mean = models.FloatField()
    increase = models.BooleanField()
    variance = models.FloatField()
    iterations = models.IntegerField()
    start_date = models.CharField(max_length=255)
    forecast_start_date = models.CharField(max_length=255)
    end_date = models.CharField(max_length=255)
    start_price = models.FloatField()
    forecast_median = models.FloatField()
    end_price = models.FloatField()
    fcst_start_price = models.FloatField()
    probability_of_positive_return = models.FloatField()
    id = models.AutoField(primary_key=True)

    class Meta:
        db_table = 'dim_tickers_stocks_prc_qnt_gbm_fcst'
        managed = False
        constraints = [
            models.UniqueConstraint(['symbol', 'date'], name='unique_dim_tickers_stocks_prc_qnt_gbm_fcst')
        ]
