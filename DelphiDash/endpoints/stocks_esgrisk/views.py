from django.shortcuts import render

# Create your views here.
from .models import *
from .serializers import *
from rest_framework import viewsets
from rest_framework.response import Response

class StocksEsgriskViewSet(viewsets.ModelViewSet):
    serializer_class = StocksEsgriskSerializer
    queryset = StocksEsgrisk.objects.all()

    def list(self, request):
        queryset = StocksEsgrisk.objects.all()
        serializer = StocksEsgriskSerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        queryset = StocksEsgrisk.objects.filter(symbol=pk)
        serializer = StocksEsgriskSerializer(queryset, many=True)
        return Response(serializer.data)