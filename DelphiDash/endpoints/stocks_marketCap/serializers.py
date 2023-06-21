from rest_framework import serializers
from .models import *

class StocksMarketCapSerializer(serializers.ModelSerializer):
    class Meta:
        model = StocksMarketCap
        fields = ('date', 'symbol', 'market_cap',)