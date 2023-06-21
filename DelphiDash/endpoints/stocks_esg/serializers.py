from rest_framework import serializers
from .models import *

class StocksEsgSerializer(serializers.ModelSerializer):
    class Meta:
        model = StocksEsg
        fields = ('symbol', 'date', 'cik', 'company_name', 'form_type', 'accepted_date', 'environmental_score', 'social_score', 'governance_score', 'esg_score', 'url',)