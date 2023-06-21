from django.shortcuts import render

# Create your views here.
from .models import *
from .serializers import *
from rest_framework import viewsets
from rest_framework.response import Response

class StocksDcfDailyViewSet(viewsets.ModelViewSet):
    serializer_class = StocksDcfDailySerializer
    queryset = StocksDcfDaily.objects.all()

    def list(self, request):
        queryset = StocksDcfDaily.objects.all()
        serializer = StocksDcfDailySerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        queryset = StocksDcfDaily.objects.filter(symbol=pk)
        serializer = StocksDcfDailySerializer(queryset, many=True)
        return Response(serializer.data)