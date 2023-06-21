from rest_framework import serializers
from .models import *

class StocksHouseTradesSerializer(serializers.ModelSerializer):
    class Meta:
        model = StocksHouseTrades
        fields = ('symbol', 'date', 'representative', 'transaction', 'amount', 'range',)