from rest_framework import serializers
from .models import *

class StocksCongresstradesSerializer(serializers.ModelSerializer):
    class Meta:
        model = StocksCongresstrades
        fields = ('report_date', 'transaction_date', 'symbol', 'representative', 'transaction', 'amount', 'house', 'range',)