from rest_framework import serializers
from .models import *

class StocksEarningsCallsInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = StocksEarningsCallsInfo
        fields = ('date', 'symbol', 'year', 'earnings_date',)