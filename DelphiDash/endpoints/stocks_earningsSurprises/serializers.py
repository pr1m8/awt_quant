from rest_framework import serializers
from .models import *

class StocksEarningsSurprisesSerializer(serializers.ModelSerializer):
    class Meta:
        model = StocksEarningsSurprises
        fields = ('date', 'symbol', 'actual_earning_result', 'estimated_earning',)