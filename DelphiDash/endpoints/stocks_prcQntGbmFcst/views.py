from django.shortcuts import render

# Create your views here.
from .models import *
from .serializers import *
from rest_framework import viewsets
from rest_framework.response import Response

class StocksPrcQntGbmFcstViewSet(viewsets.ModelViewSet):
    serializer_class = StocksPrcQntGbmFcstSerializer
    queryset = StocksPrcQntGbmFcst.objects.all()

    def list(self, request):
        queryset = StocksPrcQntGbmFcst.objects.all()
        serializer = StocksPrcQntGbmFcstSerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        queryset = StocksPrcQntGbmFcst.objects.filter(symbol=pk)
        serializer = StocksPrcQntGbmFcstSerializer(queryset, many=True)
        return Response(serializer.data)