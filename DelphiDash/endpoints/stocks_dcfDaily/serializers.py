from rest_framework import serializers
from .models import *

class StocksDcfDailySerializer(serializers.ModelSerializer):
    class Meta:
        model = StocksDcfDaily
        fields = ('date', 'symbol', 'dcf',)