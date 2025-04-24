#ENUMs

to_Bin = ['last_private', 'last_watchlist_private']
to_Prsnt = ['phone_filtered_id', 'phone_watchlisted_id']

to_Dummy = ['last_type',
'last_device_type',
'primary_banner_view_first_touch_device_type',
'first_device_type',
'primary_banner_view_last_touch_device_type',
'first_type']

to_freq = ['last_accessed_from',
'secondary_banner_view_last_touch_banner_title',
'primary_banner_view_last_touch_banner_title',
'primary_banner_view_first_touch_banner_title',
'first_accessed_from',
'secondary_banner_view_first_touch_banner_title',
'last_scid',
'first_scid',
'last_watchlist_scid',
'manager_views_last_touch_display_name']

to_num = ['Days_ECO','count_sc_views_fee_model_based',
 'count_sc_views_fee',
 'last_hours_difference',
 'stocks_invested_amount',
 'smallcap_aum',
 'etf_invested_amount',
 'midcap_aum',
 'primary_banner_view_last_touch_hours_taken_from_plift',
 'total_sid_count',
 'count_sc_views',
 'count_watchlisted_fee',
 'total_sc_buy_invested_amount',
 'count_manager_views',
 'count_sc_views_free',
 'count_sc_views_fee_sector_trackers',
 'manager_views_last_touch_hours_taken_from_plift',
 'count_sc_views_free_etf',
 'count_primary_banner_views_category_track',
 'count_primary_banner_views_category_others',
 'count_primary_banner_views_category_lamf',
 'count_sc_views_accessed_search',
 'secondary_banner_view_last_touch_hours_taken_from_plift',
 'count_primary_banner_views_android',
 'count_secondary_banner_views_category_lamf',
 'count_primary_banner_views',
 'count_primary_banner_views_category_brand_campaign_offer',
 'count_primary_banner_views_category_buy_or_fp',
 'last_filtered_hours_difference',
 'count_manager_views_name_others',
 'count_sc_views_android',
 'count_primary_banner_views_ios',
 'count_sc_views_accessed_explore',
 'count_primary_banner_views_category_subs',
 'last_watchlist_hours_difference',
 'count_primary_banner_views_category_brand_campaign',
 'count_manager_views_accessed_smallcase_profile',
 'count_sc_views_accessed_home',
 'count_sc_views_free_model_based',
 'count_primary_banner_views_category_app_referral',
 'count_filtered',
 'first_filtered_hours_difference',
 'count_primary_banner_views_web',
 'first_hours_difference',
 'count_sc_views_web',
 'count_sc_views_accessed_watchlist',
 'count_sc_views_free_sector_trackers',
 'first_watchlist_hours_difference',
 'count_manager_views_android',
 'count_sc_views_free_awi',
 'count_watchlisted',
 'count_secondary_banner_views',
 'sc_created_last_touch_hours_taken_from_plift']

set1 = set(to_Dummy)
set2 = set(to_freq)
set3 = set(to_Bin)
set4 = set(to_Prsnt)
# set5 = set(to_Sub)
set6 = set(to_num)

# Combine the sets using union
combined_set = set1 | set2 | set3 | set4 | set6