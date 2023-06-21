from django.shortcuts import render

# Create your views here.
from .models import *
from .serializers import *
from rest_framework import viewsets
from rest_framework.response import Response

class StocksGradesViewSet(viewsets.ModelViewSet):
    serializer_class = StocksGradesSerializer
    queryset = StocksGrades.objects.all()

    def list(self, request):
        queryset = StocksGrades.objects.all()
        serializer = StocksGradesSerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        queryset = StocksGrades.objects.filter(symbol=pk)
        serializer = StocksGradesSerializer(queryset, many=True)
        return Response(serializer.data)