from django.db import models

# Create your models here.
class Stockshstemployees(models.Model):
    symbol = models.CharField(max_length=255)
    cik = models.CharField(max_length=255)
    acceptanceTime = models.CharField(max_length=255)
    periodOfReport = models.CharField(max_length=255)
    companyName = models.CharField(max_length=255)
    formType = models.CharField(max_length=255)
    filingDate = models.CharField(max_length=255)
    employeeCount = models.IntegerField()
    source = models.CharField(max_length=255)
    symbol = models.CharField(max_length=10,primary_key=True)

    class Meta:
        db_table = 'dim_ticker_stocks_hst_employees_fmp'
        managed = False
        constraints = [
            models.UniqueConstraint(['symbol', 'date'], name='unique_dim_ticker_stocks_hst_employees_fmp')
        ]
