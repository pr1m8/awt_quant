from django.shortcuts import render

# Create your views here.
from .models import *
from .serializers import *
from rest_framework import viewsets
from rest_framework.response import Response

class StocksDcfAdvancedLeveredWaccViewSet(viewsets.ModelViewSet):
    serializer_class = StocksDcfAdvancedLeveredWaccSerializer
    queryset = StocksDcfAdvancedLeveredWacc.objects.all()

    def list(self, request):
        queryset = StocksDcfAdvancedLeveredWacc.objects.all()
        serializer = StocksDcfAdvancedLeveredWaccSerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        queryset = StocksDcfAdvancedLeveredWacc.objects.filter(symbol=pk)
        serializer = StocksDcfAdvancedLeveredWaccSerializer(queryset, many=True)
        return Response(serializer.data)