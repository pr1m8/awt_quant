from endpoints.stocks_tickersStocksEsg.views import StocksTickersStocksEsgViewSet
from endpoints.stocks_earningsSurprises.views import StocksEarningsSurprisesViewSet
from endpoints.stocks_earningsCallsInfo.views import StocksEarningsCallsInfoViewSet
from endpoints.stocks_dcfDaily.views import StocksDcfDailyViewSet
from endpoints.stocks_dcfAdvancedWaccProj.views import StocksDcfAdvancedWaccProjViewSet
from endpoints.stocks_dcfAdvancedLeveredWacc.views import StocksDcfAdvancedLeveredWaccViewSet
from endpoints.stocks_corporatelobbying.views import StocksCorporatelobbyingViewSet
from endpoints.stocks_congresstrades.views import StocksCongresstradesViewSet
from endpoints.stocks_companyNotes.views import StocksCompanyNotesViewSet
from endpoints.stocks_acquisitionOfBeneficialOwnership.views import StocksAcquisitionOfBeneficialOwnershipViewSet
from endpoints.stocks_esgrisk.views import StocksEsgriskViewSet
from endpoints.stocks_ratings.views import StocksRatingsViewSet
from endpoints.stocks_employees.views import StocksEmployeesViewSet
from endpoints.stocks_upgradedowngrade.views import StocksUpgradedowngradeViewSet
from endpoints.stocks_tradingSenateQvr.views import StocksTradingSenateQvrViewSet
from endpoints.stocks_tradingHouseQvr.views import StocksTradingHouseQvrViewSet
from endpoints.stocks_prcQntGbmFcst.views import StocksPrcQntGbmFcstViewSet
from endpoints.stocks_offExchange.views import StocksOffExchangeViewSet
from endpoints.stocks_prctarget.views import StocksPrctargetViewSet
from endpoints.stocks_keyExecutives.views import StocksKeyExecutivesViewSet
from endpoints.stocks_houseTrades.views import StocksHouseTradesViewSet
from endpoints.stocks_insidertrades.views import StocksInsidertradesViewSet
from endpoints.stocks_hstPrcDailyYf.views import StocksHstPrcDailyYfViewSet
from endpoints.stocks_grades.views import StocksGradesViewSet
from endpoints.stocks_fundFs.views import StocksFundFsViewSet
from endpoints.stocks_esg.views import StocksEsgViewSet
from endpoints.stocks_corporatelobbyingQvr.views import StocksCorporatelobbyingQvrViewSet
from endpoints.stocks_prcDailyAll.views import StocksPrcDailyAllViewSet
from endpoints.stocks_marketCap.views import StocksMarketCapViewSet
from endpoints.stocks_earningsCalls.views import StocksEarningsCallsViewSet
from endpoints.stocks_analystEstimates.views import StocksAnalystEstimatesViewSet
from endpoints.stocks_press.views import StocksPressViewSet
from endpoints.stocks_prcDaily.views import StocksPrcDailyViewSet
from endpoints.countries_marketRiskPremium.views import CountriesMarketRiskPremiumViewSet
from endpoints.stocks_hstEmployees.views import StocksHstEmployeesViewSet

from django.contrib import admin
from django.urls import path, include

from rest_framework import routers
router = routers.DefaultRouter()
router.register(r'StocksTickersStocksEsg', StocksTickersStocksEsgViewSet, 'StocksTickersStocksEsg')
router.register(r'StocksEarningsSurprises', StocksEarningsSurprisesViewSet, 'StocksEarningsSurprises')
router.register(r'StocksEarningsCallsInfo', StocksEarningsCallsInfoViewSet, 'StocksEarningsCallsInfo')
router.register(r'StocksDcfDaily', StocksDcfDailyViewSet, 'StocksDcfDaily')
router.register(r'StocksDcfAdvancedWaccProj', StocksDcfAdvancedWaccProjViewSet, 'StocksDcfAdvancedWaccProj')
router.register(r'StocksDcfAdvancedLeveredWacc', StocksDcfAdvancedLeveredWaccViewSet, 'StocksDcfAdvancedLeveredWacc')
router.register(r'StocksCorporatelobbying', StocksCorporatelobbyingViewSet, 'StocksCorporatelobbying')
router.register(r'StocksCongresstrades', StocksCongresstradesViewSet, 'StocksCongresstrades')
router.register(r'StocksCompanyNotes', StocksCompanyNotesViewSet, 'StocksCompanyNotes')
router.register(r'StocksAcquisitionOfBeneficialOwnership', StocksAcquisitionOfBeneficialOwnershipViewSet, 'StocksAcquisitionOfBeneficialOwnership')
router.register(r'StocksEsgrisk', StocksEsgriskViewSet, 'StocksEsgrisk')
router.register(r'StocksRatings', StocksRatingsViewSet, 'StocksRatings')
router.register(r'StocksPress', StocksPressViewSet, 'StocksPress')
router.register(r'StocksEmployees', StocksEmployeesViewSet, 'StocksEmployees')
router.register(r'StocksUpgradedowngrade', StocksUpgradedowngradeViewSet, 'StocksUpgradedowngrade')
router.register(r'StocksTradingSenateQvr', StocksTradingSenateQvrViewSet, 'StocksTradingSenateQvr')
router.register(r'StocksTradingHouseQvr', StocksTradingHouseQvrViewSet, 'StocksTradingHouseQvr')
router.register(r'StocksPrcQntGbmFcst', StocksPrcQntGbmFcstViewSet, 'StocksPrcQntGbmFcst')
router.register(r'StocksOffExchange', StocksOffExchangeViewSet, 'StocksOffExchange')
router.register(r'StocksPrctarget', StocksPrctargetViewSet, 'StocksPrctarget')
router.register(r'StocksKeyExecutives', StocksKeyExecutivesViewSet, 'StocksKeyExecutives')
router.register(r'StocksHouseTrades', StocksHouseTradesViewSet, 'StocksHouseTrades')
router.register(r'StocksInsidertrades', StocksInsidertradesViewSet, 'StocksInsidertrades')
router.register(r'StocksHstPrcDailyYf', StocksHstPrcDailyYfViewSet, 'StocksHstPrcDailyYf')
router.register(r'StocksGrades', StocksGradesViewSet, 'StocksGrades')
router.register(r'StocksFundFs', StocksFundFsViewSet, 'StocksFundFs')
router.register(r'StocksEsg', StocksEsgViewSet, 'StocksEsg')
router.register(r'StocksCorporatelobbyingQvr', StocksCorporatelobbyingQvrViewSet, 'StocksCorporatelobbyingQvr')
router.register(r'StocksPrcDailyAll', StocksPrcDailyAllViewSet, 'StocksPrcDailyAll')
router.register(r'StocksMarketCap', StocksMarketCapViewSet, 'StocksMarketCap')
router.register(r'StocksEarningsCalls', StocksEarningsCallsViewSet, 'StocksEarningsCalls')
router.register(r'StocksAnalystEstimates', StocksAnalystEstimatesViewSet, 'StocksAnalystEstimates')
router.register(r'StocksPrcDaily', StocksPrcDailyViewSet, 'StocksPrcDaily')
router.register(r'StocksHstEmployees', StocksHstEmployeesViewSet, 'StocksHstEmployees')
router.register(r'CountriesMarketRiskPremium', CountriesMarketRiskPremiumViewSet, 'CountriesMarketRiskPremium')

urlpatterns = [ 
    path('admin/', admin.site.urls),
    path('api/', include(router.urls))
]
