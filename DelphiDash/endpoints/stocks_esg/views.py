from django.shortcuts import render

# Create your views here.
from .models import *
from .serializers import *
from rest_framework import viewsets
from rest_framework.response import Response

class StocksEsgViewSet(viewsets.ModelViewSet):
    serializer_class = StocksEsgSerializer
    queryset = StocksEsg.objects.all()

    def list(self, request):
        queryset = StocksEsg.objects.all()
        serializer = StocksEsgSerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        queryset = StocksEsg.objects.filter(symbol=pk)
        serializer = StocksEsgSerializer(queryset, many=True)
        return Response(serializer.data)