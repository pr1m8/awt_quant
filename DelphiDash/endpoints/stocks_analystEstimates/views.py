from django.shortcuts import render

# Create your views here.
from .models import *
from .serializers import *
from rest_framework import viewsets
from rest_framework.response import Response

class StocksAnalystEstimatesViewSet(viewsets.ModelViewSet):
    serializer_class = StocksAnalystEstimatesSerializer
    queryset = StocksAnalystEstimates.objects.all()

    def list(self, request):
        queryset = StocksAnalystEstimates.objects.all()
        serializer = StocksAnalystEstimatesSerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        queryset = StocksAnalystEstimates.objects.filter(symbol=pk)
        serializer = StocksAnalystEstimatesSerializer(queryset, many=True)
        return Response(serializer.data)