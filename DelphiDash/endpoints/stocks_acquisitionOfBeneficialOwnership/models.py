from django.db import models

# Create your models here.
from django.db import models

class StocksAcquisitionOfBeneficialOwnership(models.Model):
    symbol = models.CharField(max_length=255)
    cik = models.IntegerField()
    filing_date = models.CharField(max_length=255)
    accepted_date = models.CharField(max_length=255)
    cusip = models.CharField(max_length=255)
    name_of_reporting_person = models.CharField(max_length=255)
    citizenship_or_place_of_organization = models.CharField(max_length=255)
    sole_voting_power = models.FloatField()
    shared_voting_power = models.FloatField()
    sole_dispositive_power = models.FloatField()
    shared_dispositive_power = models.FloatField()
    amount_beneficially_owned = models.FloatField()
    percent_of_class = models.FloatField()
    type_of_reporting_person = models.CharField(max_length=255)
    url = models.CharField(max_length=255)
    symbol = models.CharField(max_length=10,primary_key=True)

    class Meta:
        db_table = 'dim_tickers_stocks_acquisition_of_beneficial_ownership'
        managed = False
        constraints = [
            models.UniqueConstraint(['symbol', 'date'], name='unique_dim_tickers_stocks_acquisition_of_beneficial_ownership')
        ]
