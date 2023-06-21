from django.shortcuts import render

# Create your views here.
from .models import *
from .serializers import *
from rest_framework import viewsets
from rest_framework.response import Response

class StocksHstPrcDailyYfViewSet(viewsets.ModelViewSet):
    serializer_class = StocksHstPrcDailyYfSerializer
    queryset = StocksHstPrcDailyYf.objects.all()

    def list(self, request):
        queryset = StocksHstPrcDailyYf.objects.all()
        serializer = StocksHstPrcDailyYfSerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        queryset = StocksHstPrcDailyYf.objects.filter(symbol=pk)
        serializer = StocksHstPrcDailyYfSerializer(queryset, many=True)
        return Response(serializer.data)