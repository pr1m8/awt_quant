from django.shortcuts import render

# Create your views here.
from .models import *
from .serializers import *
from rest_framework import viewsets
from rest_framework.response import Response

class StocksOffExchangeViewSet(viewsets.ModelViewSet):
    serializer_class = StocksOffExchangeSerializer
    queryset = StocksOffExchange.objects.all()

    def list(self, request):
        queryset = StocksOffExchange.objects.all()
        serializer = StocksOffExchangeSerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        queryset = StocksOffExchange.objects.filter(symbol=pk)
        serializer = StocksOffExchangeSerializer(queryset, many=True)
        return Response(serializer.data)