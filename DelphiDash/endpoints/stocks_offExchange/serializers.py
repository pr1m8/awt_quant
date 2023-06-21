from rest_framework import serializers
from .models import *

class StocksOffExchangeSerializer(serializers.ModelSerializer):
    class Meta:
        model = StocksOffExchange
        fields = ('symbol', 'date', 'otc_short', 'otc_total', 'dpi',)