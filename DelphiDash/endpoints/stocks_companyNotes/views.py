from django.shortcuts import render

# Create your views here.
from .models import *
from .serializers import *
from rest_framework import viewsets
from rest_framework.response import Response

class StocksCompanyNotesViewSet(viewsets.ModelViewSet):
    serializer_class = StocksCompanyNotesSerializer
    queryset = StocksCompanyNotes.objects.all()

    def list(self, request):
        queryset = StocksCompanyNotes.objects.all()
        serializer = StocksCompanyNotesSerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        queryset = StocksCompanyNotes.objects.filter(symbol=pk)
        serializer = StocksCompanyNotesSerializer(queryset, many=True)
        return Response(serializer.data)