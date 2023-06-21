from django.shortcuts import render

# Create your views here.
from .models import *
from .serializers import *
from rest_framework import viewsets
from rest_framework.response import Response

class StocksPressViewSet(viewsets.ModelViewSet):
    serializer_class = StocksPressSerializer
    queryset = StocksPress.objects.all()

    def list(self, request):
        queryset = StocksPress.objects.all()
        serializer = StocksPressSerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        queryset = StocksPress.objects.filter(symbol=pk)
        serializer = StocksPressSerializer(queryset, many=True)
        return Response(serializer.data)