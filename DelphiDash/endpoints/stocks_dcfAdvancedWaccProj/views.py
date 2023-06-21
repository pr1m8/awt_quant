from django.shortcuts import render

# Create your views here.
from .models import *
from .serializers import *
from rest_framework import viewsets
from rest_framework.response import Response

class StocksDcfAdvancedWaccProjViewSet(viewsets.ModelViewSet):
    serializer_class = StocksDcfAdvancedWaccProjSerializer
    queryset = StocksDcfAdvancedWaccProj.objects.all()

    def list(self, request):
        queryset = StocksDcfAdvancedWaccProj.objects.all()
        serializer = StocksDcfAdvancedWaccProjSerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        queryset = StocksDcfAdvancedWaccProj.objects.filter(symbol=pk)
        serializer = StocksDcfAdvancedWaccProjSerializer(queryset, many=True)
        return Response(serializer.data)