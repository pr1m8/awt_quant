from django.shortcuts import render

# Create your views here.
from .models import *
from .serializers import *
from rest_framework import viewsets
from rest_framework.response import Response

class StocksInsidertradesViewSet(viewsets.ModelViewSet):
    serializer_class = StocksInsidertradesSerializer
    queryset = StocksInsidertrades.objects.all()

    def list(self, request):
        queryset = StocksInsidertrades.objects.all()
        serializer = StocksInsidertradesSerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        queryset = StocksInsidertrades.objects.filter(symbol=pk)
        serializer = StocksInsidertradesSerializer(queryset, many=True)
        return Response(serializer.data)