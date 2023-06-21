# Generated by Django 4.2.2 on 2023-06-20 05:22

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='StocksAcquisitionOfBeneficialOwnership',
            fields=[
                ('cik', models.IntegerField()),
                ('filing_date', models.CharField(max_length=255)),
                ('accepted_date', models.CharField(max_length=255)),
                ('cusip', models.CharField(max_length=255)),
                ('name_of_reporting_person', models.CharField(max_length=255)),
                ('citizenship_or_place_of_organization', models.CharField(max_length=255)),
                ('sole_voting_power', models.FloatField()),
                ('shared_voting_power', models.FloatField()),
                ('sole_dispositive_power', models.FloatField()),
                ('shared_dispositive_power', models.FloatField()),
                ('amount_beneficially_owned', models.FloatField()),
                ('percent_of_class', models.FloatField()),
                ('type_of_reporting_person', models.CharField(max_length=255)),
                ('url', models.CharField(max_length=255)),
                ('symbol', models.CharField(max_length=10, primary_key=True, serialize=False)),
            ],
            options={
                'db_table': 'dim_tickers_stocks_acquisition_of_beneficial_ownership',
                'managed': False,
            },
        ),
    ]
