from rest_framework import serializers
from .models import *

class StocksPressSerializer(serializers.ModelSerializer):
    class Meta:
        model = StocksPress
        fields = ('symbol', 'date', 'title', 'text',)