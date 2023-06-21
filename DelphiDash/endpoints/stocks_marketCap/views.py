from django.shortcuts import render

# Create your views here.
from .models import *
from .serializers import *
from rest_framework import viewsets
from rest_framework.response import Response

class StocksMarketCapViewSet(viewsets.ModelViewSet):
    serializer_class = StocksMarketCapSerializer
    queryset = StocksMarketCap.objects.all()

    def list(self, request):
        queryset = StocksMarketCap.objects.all()
        serializer = StocksMarketCapSerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        queryset = StocksMarketCap.objects.filter(symbol=pk)
        serializer = StocksMarketCapSerializer(queryset, many=True)
        return Response(serializer.data)