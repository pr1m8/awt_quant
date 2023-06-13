from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from .models import *

from rest_framework.response import Response

class StockshstemployeesView(APIView):
    def get(self, request, symbol):
        stock_p = Stockshstemployees.objects.filter(symbol=symbol).using("finance")
        print(stock_p)
        db_table= 'dim_ticker_stocks_hst_employees_fmp'
        data = []
        for item in stock_p:
            data.append({'symbol': item.symbol, 'cik': item.cik, 'acceptanceTime': item.acceptanceTime, 'periodOfReport': item.periodOfReport, 'companyName': item.companyName, 'formType': item.formType, 'filingDate': item.filingDate, 'employeeCount': item.employeeCount, 'source': item.source, })
        return Response(data)
