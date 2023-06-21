from django.shortcuts import render

# Create your views here.
from .models import *
from .serializers import *
from rest_framework import viewsets
from rest_framework.response import Response

class StocksCorporatelobbyingViewSet(viewsets.ModelViewSet):
    serializer_class = StocksCorporatelobbyingSerializer
    queryset = StocksCorporatelobbying.objects.all()

    def list(self, request):
        queryset = StocksCorporatelobbying.objects.all()
        serializer = StocksCorporatelobbyingSerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        queryset = StocksCorporatelobbying.objects.filter(symbol=pk)
        serializer = StocksCorporatelobbyingSerializer(queryset, many=True)
        return Response(serializer.data)