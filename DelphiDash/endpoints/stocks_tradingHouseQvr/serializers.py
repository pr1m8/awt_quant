from rest_framework import serializers
from .models import *

class StocksTradingHouseQvrSerializer(serializers.ModelSerializer):
    class Meta:
        model = StocksTradingHouseQvr
        fields = ('report_date', 'transaction_date', 'symbol', 'representative', 'transaction', 'amount', 'house', 'range',)