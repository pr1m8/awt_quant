from django.db import models

# Create your models here.
from django.db import models

class StocksGrades(models.Model):
    symbol = models.CharField(max_length=255)
    date = models.CharField(max_length=255)
    grading_company = models.CharField(max_length=255)
    previous_grade = models.CharField(max_length=255)
    new_grade = models.CharField(max_length=255)
    symbol = models.CharField(max_length=10,primary_key=True)

    class Meta:
        db_table = 'dim_tickers_stocks_grades'
        managed = False
        constraints = [
            models.UniqueConstraint(['symbol', 'date'], name='unique_dim_tickers_stocks_grades')
        ]
