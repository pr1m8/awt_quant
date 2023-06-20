from .models import *
from .serializers import StocksHstEmployeesSerializer
from rest_framework import viewsets
from rest_framework.response import Response


class StocksHstEmployeesViewSet(viewsets.ModelViewSet):
    serializer_class = StocksHstEmployeesSerializer
    queryset = StocksHstEmployees.objects.all()

    def list(self, request):
        queryset = StocksHstEmployees.objects.all()
        serializer = StocksHstEmployeesSerializer(queryset, many=True)
        return Response(serializer.data)
    
    def retrieve(self, request, pk=None):
        queryset = StocksHstEmployees.objects.filter(symbol=pk)
        serializer = StocksHstEmployeesSerializer(queryset, many=True)
        return Response(serializer.data)


