from django.shortcuts import render

# Create your views here.
from .models import *
from .serializers import *
from rest_framework import viewsets
from rest_framework.response import Response

class StocksEmployeesViewSet(viewsets.ModelViewSet):
    serializer_class = StocksEmployeesSerializer
    queryset = StocksEmployees.objects.all()

    def list(self, request):
        queryset = StocksEmployees.objects.all()
        serializer = StocksEmployeesSerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        queryset = StocksEmployees.objects.filter(symbol=pk)
        serializer = StocksEmployeesSerializer(queryset, many=True)
        return Response(serializer.data)