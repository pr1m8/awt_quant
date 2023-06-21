from django.shortcuts import render

# Create your views here.
from .models import *
from .serializers import *
from rest_framework import viewsets
from rest_framework.response import Response

class StocksTradingSenateQvrViewSet(viewsets.ModelViewSet):
    serializer_class = StocksTradingSenateQvrSerializer
    queryset = StocksTradingSenateQvr.objects.all()

    def list(self, request):
        queryset = StocksTradingSenateQvr.objects.all()
        serializer = StocksTradingSenateQvrSerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        queryset = StocksTradingSenateQvr.objects.filter(symbol=pk)
        serializer = StocksTradingSenateQvrSerializer(queryset, many=True)
        return Response(serializer.data)