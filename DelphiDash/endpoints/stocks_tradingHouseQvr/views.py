from django.shortcuts import render

# Create your views here.
from .models import *
from .serializers import *
from rest_framework import viewsets
from rest_framework.response import Response

class StocksTradingHouseQvrViewSet(viewsets.ModelViewSet):
    serializer_class = StocksTradingHouseQvrSerializer
    queryset = StocksTradingHouseQvr.objects.all()

    def list(self, request):
        queryset = StocksTradingHouseQvr.objects.all()
        serializer = StocksTradingHouseQvrSerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        queryset = StocksTradingHouseQvr.objects.filter(symbol=pk)
        serializer = StocksTradingHouseQvrSerializer(queryset, many=True)
        return Response(serializer.data)