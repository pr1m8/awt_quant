from django.db import models

# Create your models here.
class StocksHstEmployees(models.Model):
    cik = models.CharField(max_length=255)
    acceptance_time = models.CharField(max_length=255)
    period_of_report = models.CharField(max_length=255)
    company_name = models.CharField(max_length=255)
    form_type = models.CharField(max_length=255)
    filing_date = models.CharField(max_length=255)
    employee_count = models.IntegerField()
    source = models.CharField(max_length=255)
    symbol = models.CharField(max_length=10,primary_key=True)

    class Meta:
        db_table = 'dim_ticker_stocks_hst_employees_fmp'
        managed = False
        constraints = [
            models.UniqueConstraint(['symbol', 'filing_date'], name='unique_dim_ticker_stocks_hst_employees_fmp')
        ]
