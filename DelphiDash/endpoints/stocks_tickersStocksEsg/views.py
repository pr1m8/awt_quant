from django.shortcuts import render

# Create your views here.
from .models import *
from .serializers import *
from rest_framework import viewsets
from rest_framework.response import Response

class StocksTickersStocksEsgViewSet(viewsets.ModelViewSet):
    serializer_class = StocksTickersStocksEsgSerializer
    queryset = StocksTickersStocksEsg.objects.all()

    def list(self, request):
        queryset = StocksTickersStocksEsg.objects.all()
        serializer = StocksTickersStocksEsgSerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        queryset = StocksTickersStocksEsg.objects.filter(symbol=pk)
        serializer = StocksTickersStocksEsgSerializer(queryset, many=True)
        return Response(serializer.data)