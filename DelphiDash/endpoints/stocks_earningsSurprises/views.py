from django.shortcuts import render

# Create your views here.
from .models import *
from .serializers import *
from rest_framework import viewsets
from rest_framework.response import Response

class StocksEarningsSurprisesViewSet(viewsets.ModelViewSet):
    serializer_class = StocksEarningsSurprisesSerializer
    queryset = StocksEarningsSurprises.objects.all()

    def list(self, request):
        queryset = StocksEarningsSurprises.objects.all()
        serializer = StocksEarningsSurprisesSerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        queryset = StocksEarningsSurprises.objects.filter(symbol=pk)
        serializer = StocksEarningsSurprisesSerializer(queryset, many=True)
        return Response(serializer.data)