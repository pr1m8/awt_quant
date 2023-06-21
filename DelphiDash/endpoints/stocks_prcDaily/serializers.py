from rest_framework import serializers
from .models import *

class StocksPrcDailySerializer(serializers.ModelSerializer):
    class Meta:
        model = StocksPrcDaily
        fields = ('symbol', 'date', 'open', 'close', 'high', 'low', 'adj_close', 'volume')