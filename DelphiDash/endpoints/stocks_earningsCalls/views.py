from django.shortcuts import render

# Create your views here.
from .models import *
from .serializers import *
from rest_framework import viewsets
from rest_framework.response import Response

class StocksEarningsCallsViewSet(viewsets.ModelViewSet):
    serializer_class = StocksEarningsCallsSerializer
    queryset = StocksEarningsCalls.objects.all()

    def list(self, request):
        queryset = StocksEarningsCalls.objects.all()
        serializer = StocksEarningsCallsSerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        queryset = StocksEarningsCalls.objects.filter(symbol=pk)
        serializer = StocksEarningsCallsSerializer(queryset, many=True)
        return Response(serializer.data)