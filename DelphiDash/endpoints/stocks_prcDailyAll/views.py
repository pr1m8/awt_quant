from django.shortcuts import render

# Create your views here.
from .models import *
from .serializers import *
from rest_framework import viewsets
from rest_framework.response import Response

class StocksPrcDailyAllViewSet(viewsets.ModelViewSet):
    serializer_class = StocksPrcDailyAllSerializer
    queryset = StocksPrcDailyAll.objects.all()

    def list(self, request):
        queryset = StocksPrcDailyAll.objects.all()
        serializer = StocksPrcDailyAllSerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        queryset = StocksPrcDailyAll.objects.filter(symbol=pk)
        serializer = StocksPrcDailyAllSerializer(queryset, many=True)
        return Response(serializer.data)