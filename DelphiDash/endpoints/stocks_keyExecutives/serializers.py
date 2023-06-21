from rest_framework import serializers
from .models import *

class StocksKeyExecutivesSerializer(serializers.ModelSerializer):
    class Meta:
        model = StocksKeyExecutives
        fields = ('title', 'name', 'pay', 'currency_pay', 'gender', 'year_born', 'title_since',)