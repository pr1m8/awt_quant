from django.shortcuts import render

# Create your views here.
from .models import *
from .serializers import *
from rest_framework import viewsets
from rest_framework.response import Response

class StocksKeyExecutivesViewSet(viewsets.ModelViewSet):
    serializer_class = StocksKeyExecutivesSerializer
    queryset = StocksKeyExecutives.objects.all()

    def list(self, request):
        queryset = StocksKeyExecutives.objects.all()
        serializer = StocksKeyExecutivesSerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        queryset = StocksKeyExecutives.objects.filter(symbol=pk)
        serializer = StocksKeyExecutivesSerializer(queryset, many=True)
        return Response(serializer.data)