from django.shortcuts import render

# Create your views here.
from .models import *
from .serializers import *
from rest_framework import viewsets
from rest_framework.response import Response

class StocksCorporatelobbyingQvrViewSet(viewsets.ModelViewSet):
    serializer_class = StocksCorporatelobbyingQvrSerializer
    queryset = StocksCorporatelobbyingQvr.objects.all()

    def list(self, request):
        queryset = StocksCorporatelobbyingQvr.objects.all()
        serializer = StocksCorporatelobbyingQvrSerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        queryset = StocksCorporatelobbyingQvr.objects.filter(symbol=pk)
        serializer = StocksCorporatelobbyingQvrSerializer(queryset, many=True)
        return Response(serializer.data)