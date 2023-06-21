from rest_framework import serializers
from .models import *

class StocksPrcDailyAllSerializer(serializers.ModelSerializer):
    class Meta:
        model = StocksPrcDailyAll
        fields = ('date', 'symbol', 'open', 'high', 'low', 'close', 'adj_close', 'volume',)