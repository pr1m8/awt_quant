from django.shortcuts import render

# Create your views here.
from .models import *
from .serializers import *
from rest_framework import viewsets
from rest_framework.response import Response

class StocksCongresstradesViewSet(viewsets.ModelViewSet):
    serializer_class = StocksCongresstradesSerializer
    queryset = StocksCongresstrades.objects.all()

    def list(self, request):
        queryset = StocksCongresstrades.objects.all()
        serializer = StocksCongresstradesSerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        queryset = StocksCongresstrades.objects.filter(symbol=pk)
        serializer = StocksCongresstradesSerializer(queryset, many=True)
        return Response(serializer.data)