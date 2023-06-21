from django.shortcuts import render

# Create your views here.
from .models import *
from .serializers import *
from rest_framework import viewsets
from rest_framework.response import Response

class StocksUpgradedowngradeViewSet(viewsets.ModelViewSet):
    serializer_class = StocksUpgradedowngradeSerializer
    queryset = StocksUpgradedowngrade.objects.all()

    def list(self, request):
        queryset = StocksUpgradedowngrade.objects.all()
        serializer = StocksUpgradedowngradeSerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        queryset = StocksUpgradedowngrade.objects.filter(symbol=pk)
        serializer = StocksUpgradedowngradeSerializer(queryset, many=True)
        return Response(serializer.data)