from django.db import models

# Create your models here.
class CountriesMarketRiskPremium(models.Model):
    country = models.CharField(max_length=255, primary_key=True)
    continent = models.CharField(max_length=255)
    total_equity_risk_premium = models.FloatField()
    country_risk_premium = models.FloatField()
    datestamp = models.CharField(max_length=255) 

    class Meta:
        db_table = 'dim_countries_market_risk_premium'
        managed = False
        constraints = [
            models.UniqueConstraint(['country'], name='unique_dim_countries_market_risk_premium')
        ]
