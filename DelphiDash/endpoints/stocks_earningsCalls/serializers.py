from rest_framework import serializers
from .models import *

class StocksEarningsCallsSerializer(serializers.ModelSerializer):
    class Meta:
        model = StocksEarningsCalls
        fields = ('date', 'symbol', 'quarter', 'year', 'content',)