from django.shortcuts import render

# Create your views here.
from .models import *
from .serializers import *
from rest_framework import viewsets
from rest_framework.response import Response

class StocksEarningsCallsInfoViewSet(viewsets.ModelViewSet):
    serializer_class = StocksEarningsCallsInfoSerializer
    queryset = StocksEarningsCallsInfo.objects.all()

    def list(self, request):
        queryset = StocksEarningsCallsInfo.objects.all()
        serializer = StocksEarningsCallsInfoSerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        queryset = StocksEarningsCallsInfo.objects.filter(symbol=pk)
        serializer = StocksEarningsCallsInfoSerializer(queryset, many=True)
        return Response(serializer.data)