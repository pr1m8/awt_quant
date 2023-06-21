# Generated by Django 4.2.2 on 2023-06-20 05:22

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='StocksCongresstrades',
            fields=[
                ('report_date', models.CharField(max_length=255)),
                ('transaction_date', models.CharField(max_length=255)),
                ('representative', models.CharField(max_length=255)),
                ('transaction', models.CharField(max_length=255)),
                ('amount', models.FloatField()),
                ('house', models.CharField(max_length=255)),
                ('range', models.CharField(max_length=255)),
                ('symbol', models.CharField(max_length=10, primary_key=True, serialize=False)),
            ],
            options={
                'db_table': 'dim_tickers_stocks_congresstrades',
                'managed': False,
            },
        ),
    ]
