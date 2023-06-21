from rest_framework import serializers
from .models import *

class StocksTradingSenateQvrSerializer(serializers.ModelSerializer):
    class Meta:
        model = StocksTradingSenateQvr
        fields = ('symbol', 'date', 'senator', 'transaction', 'amount', 'range',)