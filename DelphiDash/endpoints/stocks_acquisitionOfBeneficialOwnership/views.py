from django.shortcuts import render

# Create your views here.
from .models import *
from .serializers import *
from rest_framework import viewsets
from rest_framework.response import Response

class StocksAcquisitionOfBeneficialOwnershipViewSet(viewsets.ModelViewSet):
    serializer_class = StocksAcquisitionOfBeneficialOwnershipSerializer
    queryset = StocksAcquisitionOfBeneficialOwnership.objects.all()

    def list(self, request):
        queryset = StocksAcquisitionOfBeneficialOwnership.objects.all()
        serializer = StocksAcquisitionOfBeneficialOwnershipSerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        queryset = StocksAcquisitionOfBeneficialOwnership.objects.filter(symbol=pk)
        serializer = StocksAcquisitionOfBeneficialOwnershipSerializer(queryset, many=True)
        return Response(serializer.data)