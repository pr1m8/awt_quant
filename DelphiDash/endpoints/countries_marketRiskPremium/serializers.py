from rest_framework import serializers
from .models import CountriesMarketRiskPremium 

class CountriesMarketRiskPremiumSerializer(serializers.ModelSerializer):
    class Meta:
        model = CountriesMarketRiskPremium
        fields = ('country', 'continent', 'total_equity_risk_premium', 'country_risk_premium', 'datestamp')
                 
