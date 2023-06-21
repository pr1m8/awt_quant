from rest_framework import serializers
from .models import *

class StocksEsgriskSerializer(serializers.ModelSerializer):
    class Meta:
        model = StocksEsgrisk
        fields = ('symbol', 'date', 'cik', 'company_name', 'form_type', 'accepted_date', 'environmental_score', 'social_score', 'governance_score', 'esg_score', 'url',)