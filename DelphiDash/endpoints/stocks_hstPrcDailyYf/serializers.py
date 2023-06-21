from rest_framework import serializers
from .models import *

class StocksHstPrcDailyYfSerializer(serializers.ModelSerializer):
    class Meta:
        model = StocksHstPrcDailyYf
        fields = ('date', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits', 'capital_gains', 'symbol',)