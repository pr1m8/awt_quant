from django.db import models

# Create your models here.
from django.db import models

class StocksEmployees(models.Model):
    symbol = models.CharField(max_length=255)
    cik = models.CharField(max_length=255)
    filing_date = models.CharField(max_length=255)
    acceptance_time = models.CharField(max_length=255)
    period_of_report = models.CharField(max_length=255)
    company_name = models.CharField(max_length=255)
    form_type = models.CharField(max_length=255)
    employee_count = models.IntegerField()
    source = models.CharField(max_length=255)
    symbol = models.CharField(max_length=10,primary_key=True)

    class Meta:
        db_table = 'dim_tickers_stocks_employees'
        managed = False
        constraints = [
            models.UniqueConstraint(['symbol', 'date'], name='unique_dim_tickers_stocks_employees')
        ]
