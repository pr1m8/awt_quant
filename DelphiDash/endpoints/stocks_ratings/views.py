from django.shortcuts import render

# Create your views here.
from .models import *
from .serializers import *
from rest_framework import viewsets
from rest_framework.response import Response

class StocksRatingsViewSet(viewsets.ModelViewSet):
    serializer_class = StocksRatingsSerializer
    queryset = StocksRatings.objects.all()

    def list(self, request):
        queryset = StocksRatings.objects.all()
        serializer = StocksRatingsSerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        queryset = StocksRatings.objects.filter(symbol=pk)
        serializer = StocksRatingsSerializer(queryset, many=True)
        return Response(serializer.data)