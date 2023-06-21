from rest_framework import serializers
from .models import *

class StocksCorporatelobbyingQvrSerializer(serializers.ModelSerializer):
    class Meta:
        model = StocksCorporatelobbyingQvr
        fields = ('symbol', 'date', 'client', 'amount', 'issue', 'specific_issue',)