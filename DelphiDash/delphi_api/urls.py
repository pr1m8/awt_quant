from endpoints.countries_marketRiskPremium.views import CountriesMarketRiskPremiumViewSet
from endpoints.stocks_hstEmployees.views import StocksHstEmployeesViewSet

from django.contrib import admin
from django.urls import path, include

from rest_framework import routers
router = routers.DefaultRouter()
router.register(r'StocksHstEmployees', StocksHstEmployeesViewSet, 'StocksHstEmployees')
router.register(r'CountriesMarketRiskPremium', CountriesMarketRiskPremiumViewSet, 'CountriesMarketRiskPremium')

urlpatterns = [ 
    path('admin/', admin.site.urls),
    path('api/', include(router.urls))
]
