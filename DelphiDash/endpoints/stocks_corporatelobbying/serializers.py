from rest_framework import serializers
from .models import *

class StocksCorporatelobbyingSerializer(serializers.ModelSerializer):
    class Meta:
        model = StocksCorporatelobbying
        fields = ('date', 'symbol', 'client', 'amount', 'issue', 'specific_issue', 'registrant',)