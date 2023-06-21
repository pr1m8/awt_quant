from .models import *
from .serializers import StocksPrcDailySerializer
from rest_framework import viewsets
from rest_framework.response import Response

class StocksPrcDailyViewSet(viewsets.ModelViewSet):
    serializer_class = StocksPrcDailySerializer
    queryset = StocksPrcDaily.objects.all()
    

    def list(self, request):
        queryset = StocksPrcDaily.objects.all()
        serializer = StocksPrcDailySerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        queryset = StocksPrcDaily.objects.all().filter(symbol=pk)
        serializer = StocksPrcDailySerializer(queryset, many=True)
        return Response(serializer.data)