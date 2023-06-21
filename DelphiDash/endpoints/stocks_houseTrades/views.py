from django.shortcuts import render

# Create your views here.
from .models import *
from .serializers import *
from rest_framework import viewsets
from rest_framework.response import Response

class StocksHouseTradesViewSet(viewsets.ModelViewSet):
    serializer_class = StocksHouseTradesSerializer
    queryset = StocksHouseTrades.objects.all()

    def list(self, request):
        queryset = StocksHouseTrades.objects.all()
        serializer = StocksHouseTradesSerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        queryset = StocksHouseTrades.objects.filter(symbol=pk)
        serializer = StocksHouseTradesSerializer(queryset, many=True)
        return Response(serializer.data)