from .models import *
from .serializers import CountriesMarketRiskPremiumSerializer
from rest_framework import viewsets
from rest_framework.response import Response


class CountriesMarketRiskPremiumViewSet(viewsets.ModelViewSet):
    serializer_class = CountriesMarketRiskPremiumSerializer
    queryset = CountriesMarketRiskPremium.objects.all()

    def list(self, request):
        queryset = CountriesMarketRiskPremium.objects.all()
        serializer = CountriesMarketRiskPremiumSerializer(queryset, many=True)
        return Response(serializer.data)
    
    def retrieve(self, request, pk=None):
        queryset = CountriesMarketRiskPremium.objects.get(country=pk)
        serializer = CountriesMarketRiskPremiumSerializer(queryset, many=False)
        return Response(serializer.data)
