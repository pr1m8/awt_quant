from django.shortcuts import render

# Create your views here.
from .models import *
from .serializers import *
from rest_framework import viewsets
from rest_framework.response import Response

class StocksPrctargetViewSet(viewsets.ModelViewSet):
    serializer_class = StocksPrctargetSerializer
    queryset = StocksPrctarget.objects.all()

    def list(self, request):
        queryset = StocksPrctarget.objects.all()
        serializer = StocksPrctargetSerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        queryset = StocksPrctarget.objects.filter(symbol=pk)
        serializer = StocksPrctargetSerializer(queryset, many=True)
        return Response(serializer.data)