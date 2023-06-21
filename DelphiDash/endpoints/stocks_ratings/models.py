from django.db import models

# Create your models here.
from django.db import models

class StocksRatings(models.Model):
    symbol = models.CharField(max_length=255)
    date = models.CharField(max_length=255)
    rating = models.CharField(max_length=255)
    rating_score = models.IntegerField()
    rating_recommendation = models.CharField(max_length=255)
    rating_details_dcf_score = models.IntegerField()
    rating_details_dcf_recommendation = models.CharField(max_length=255)
    rating_details_roe_score = models.IntegerField()
    rating_details_roe_recommendation = models.CharField(max_length=255)
    rating_details_roa_score = models.IntegerField()
    rating_details_roa_recommendation = models.CharField(max_length=255)
    rating_details_de_score = models.IntegerField()
    rating_details_de_recommendation = models.CharField(max_length=255)
    rating_details_pe_score = models.IntegerField()
    rating_details_pe_recommendation = models.CharField(max_length=255)
    rating_details_pb_score = models.IntegerField()
    rating_details_pb_recommendation = models.CharField(max_length=255)
    symbol = models.CharField(max_length=10,primary_key=True)

    class Meta:
        db_table = 'dim_tickers_stocks_ratings'
        managed = False
        constraints = [
            models.UniqueConstraint(['symbol', 'date'], name='unique_dim_tickers_stocks_ratings')
        ]
