from django.shortcuts import render

# Create your views here.
from .models import *
from .serializers import *
from rest_framework import viewsets
from rest_framework.response import Response

class StocksFundFsViewSet(viewsets.ModelViewSet):
    serializer_class = StocksFundFsSerializer
    queryset = StocksFundFs.objects.all()

    def list(self, request):
        queryset = StocksFundFs.objects.all()
        serializer = StocksFundFsSerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        queryset = StocksFundFs.objects.filter(symbol=pk)
        serializer = StocksFundFsSerializer(queryset, many=True)
        return Response(serializer.data)