from django.db import models

# Create your models here.
from django.db import models

class StocksFundFs(models.Model):
    symbol = models.CharField(max_length=255)
    calendar_year = models.IntegerField()
    period = models.CharField(max_length=255)
    cik = models.FloatField()
    reported_currency = models.CharField(max_length=255)
    date = models.CharField(max_length=255)
    link = models.CharField(max_length=255)
    final_link = models.CharField(max_length=255)
    is_cost_and_expenses = models.FloatField()
    is_cost_of_revenue = models.FloatField()
    is_depreciation_and_amortization = models.FloatField()
    is_ebitda = models.FloatField()
    is_ebitdaratio = models.FloatField()
    is_eps = models.FloatField()
    is_epsdiluted = models.FloatField()
    is_general_and_administrative_expenses = models.FloatField()
    is_gross_profit = models.FloatField()
    is_gross_profit_ratio = models.FloatField()
    is_income_before_tax = models.FloatField()
    is_income_before_tax_ratio = models.FloatField()
    is_income_tax_expense = models.FloatField()
    is_interest_expense = models.FloatField()
    is_interest_income = models.FloatField()
    is_net_income = models.FloatField()
    is_net_income_ratio = models.FloatField()
    is_operating_expenses = models.FloatField()
    is_operating_income = models.FloatField()
    is_operating_income_ratio = models.FloatField()
    is_other_expenses = models.FloatField()
    is_research_and_development_expenses = models.FloatField()
    is_revenue = models.FloatField()
    is_selling_and_marketing_expenses = models.FloatField()
    is_selling_general_and_administrative_expenses = models.FloatField()
    is_total_other_income_expenses_net = models.FloatField()
    is_weighted_average_shs_out = models.FloatField()
    is_weighted_average_shs_out_dil = models.FloatField()
    bs_account_payables = models.FloatField()
    bs_accumulated_other_comprehensive_income_loss = models.FloatField()
    bs_capital_lease_obligations = models.FloatField()
    bs_cash_and_cash_equivalents = models.FloatField()
    bs_cash_and_short_term_investments = models.FloatField()
    bs_common_stock = models.FloatField()
    bs_deferred_revenue = models.FloatField()
    bs_deferred_revenue_non_current = models.FloatField()
    bs_deferred_tax_liabilities_non_current = models.FloatField()
    bs_goodwill = models.FloatField()
    bs_goodwill_and_intangible_assets = models.FloatField()
    bs_intangible_assets = models.FloatField()
    bs_inventory = models.FloatField()
    bs_long_term_debt = models.FloatField()
    bs_long_term_investments = models.FloatField()
    bs_minority_interest = models.FloatField()
    bs_net_debt = models.FloatField()
    bs_net_receivables = models.FloatField()
    bs_other_assets = models.FloatField()
    bs_other_current_assets = models.FloatField()
    bs_other_current_liabilities = models.FloatField()
    bs_other_liabilities = models.FloatField()
    bs_other_non_current_assets = models.FloatField()
    bs_other_non_current_liabilities = models.FloatField()
    bs_othertotal_stockholders_equity = models.FloatField()
    bs_preferred_stock = models.FloatField()
    bs_property_plant_equipment_net = models.FloatField()
    bs_retained_earnings = models.FloatField()
    bs_short_term_debt = models.FloatField()
    bs_short_term_investments = models.FloatField()
    bs_tax_assets = models.FloatField()
    bs_tax_payables = models.FloatField()
    bs_total_assets = models.FloatField()
    bs_total_current_assets = models.FloatField()
    bs_total_current_liabilities = models.FloatField()
    bs_total_debt = models.FloatField()
    bs_total_equity = models.FloatField()
    bs_total_investments = models.FloatField()
    bs_total_liabilities = models.FloatField()
    bs_total_liabilities_and_stockholders_equity = models.FloatField()
    bs_total_liabilities_and_total_equity = models.FloatField()
    bs_total_non_current_assets = models.FloatField()
    bs_total_non_current_liabilities = models.FloatField()
    bs_total_stockholders_equity = models.FloatField()
    cf_accounts_payables = models.FloatField()
    cf_accounts_receivables = models.FloatField()
    cf_acquisitions_net = models.FloatField()
    cf_capital_expenditure = models.FloatField()
    cf_cash_at_beginning_of_period = models.FloatField()
    cf_cash_at_end_of_period = models.FloatField()
    cf_change_in_working_capital = models.FloatField()
    cf_common_stock_issued = models.FloatField()
    cf_common_stock_repurchased = models.FloatField()
    cf_debt_repayment = models.FloatField()
    cf_deferred_income_tax = models.FloatField()
    cf_depreciation_and_amortization = models.FloatField()
    cf_dividends_paid = models.FloatField()
    cf_effect_of_forex_changes_on_cash = models.FloatField()
    cf_free_cash_flow = models.FloatField()
    cf_inventory = models.FloatField()
    cf_investments_in_property_plant_and_equipment = models.FloatField()
    cf_net_cash_provided_by_operating_activities = models.FloatField()
    cf_net_cash_used_for_investing_activites = models.FloatField()
    cf_net_cash_used_provided_by_financing_activities = models.FloatField()
    cf_net_change_in_cash = models.FloatField()
    cf_net_income = models.FloatField()
    cf_operating_cash_flow = models.FloatField()
    cf_other_financing_activites = models.FloatField()
    cf_other_investing_activites = models.FloatField()
    cf_other_non_cash_items = models.FloatField()
    cf_other_working_capital = models.FloatField()
    cf_purchases_of_investments = models.FloatField()
    cf_sales_maturities_of_investments = models.FloatField()
    cf_stock_based_compensation = models.FloatField()
    symbol = models.CharField(max_length=10,primary_key=True)

    class Meta:
        db_table = 'dim_tickers_stocks_fund_fs'
        managed = False
        constraints = [
            models.UniqueConstraint(['symbol', 'date'], name='unique_dim_tickers_stocks_fund_fs')
        ]
