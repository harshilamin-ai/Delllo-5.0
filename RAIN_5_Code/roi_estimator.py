from __future__ import annotations
from typing import Any, Dict
from .utils import safe_float

def first_product(profile:Dict[str,Any])->Dict[str,Any]:
    pricing=profile.get('commercial_pricing_json') or {}; products=pricing.get('products',[]) if isinstance(pricing,dict) else []
    return products[0] if products and isinstance(products[0],dict) else {}
def product_price_usd(profile, rate):
    p=first_product(profile); gbp=max(safe_float(p.get('enterprise_price_gbp')),safe_float(p.get('target_price_gbp')),safe_float(p.get('min_price_gbp')),safe_float(p.get('enterprise_annual_price_gbp')),safe_float(p.get('typical_annual_price_gbp'))); return gbp*rate
def product_value_case_usd(profile, rate):
    best=0.0
    for c in profile.get('product_value_cases_json') or []:
        if isinstance(c,dict): best=max(best,safe_float(c.get('estimated_annual_value_gbp')))
    return best*rate
def buyer_pain_usd(profile, rate):
    best=0.0
    for p in profile.get('buyer_pain_points_json') or []:
        if isinstance(p,dict): best=max(best,safe_float(p.get('estimated_cost_of_inaction_gbp')))
    return best*rate
def recruitment_value_usd(a,b,rate):
    for p in [a,b]:
        rec=(p.get('commercial_pricing_json') or {}).get('recruitment') if isinstance(p.get('commercial_pricing_json'),dict) else {}
        if isinstance(rec,dict):
            salary=safe_float(rec.get('typical_salary_gbp')); fee=safe_float(rec.get('recruiter_fee_pct'))
            if salary and fee: return salary*fee*rate
    return 90000*0.20*rate
def capital_value_usd(a,b,rate):
    for p in [a,b]:
        cap=(p.get('commercial_pricing_json') or {}).get('capital') if isinstance(p.get('commercial_pricing_json'),dict) else {}
        if isinstance(cap,dict):
            amount=safe_float(cap.get('typical_capital_raise_gbp')); fee=safe_float(cap.get('success_fee_pct'))
            if amount and fee: return amount*fee*rate
    return 1000000*0.02*rate

def estimate_roi_usd(person_a:Dict[str,Any], person_b:Dict[str,Any], bvh_type:Dict[str,Any], hvti_result:Dict[str,Any], hvti_structure:Dict[str,Any], calibration:Dict[str,Any])->Dict[str,Any]:
    rate=float(hvti_structure.get('value_model',{}).get('default_gbp_to_usd',1.27)); bvh_id=bvh_type['BVHTypeID']
    a_product=product_price_usd(person_a,rate); b_product=product_price_usd(person_b,rate); a_case=product_value_case_usd(person_a,rate); b_case=product_value_case_usd(person_b,rate); a_pain=buyer_pain_usd(person_a,rate); b_pain=buyer_pain_usd(person_b,rate)
    method='profile_value_signals'; candidates=[]
    if bvh_id in ['PRODUCTSALE','SUPPLIERMATCH','PROBLEMSOLUTION']:
        candidates=[a_product,a_case,b_pain]; method='source_product_price_or_target_pain'
    elif bvh_id=='CLIENTINTRODUCTION':
        candidates=[a_product*.20,b_product*.20,a_case*.15,b_case*.15]; method='estimated_referral_or_route_to_market_value'
    elif bvh_id=='CAPITALINTRODUCTION': candidates=[capital_value_usd(person_a,person_b,rate)]; method='capital_amount_success_fee'
    elif bvh_id=='PARTNERSHIP': candidates=[max(a_case,a_product)+max(b_case,b_product)]; method='combined_joint_revenue_or_value_case'
    elif bvh_id=='RECRUITMENT': candidates=[recruitment_value_usd(person_a,person_b,rate)]; method='candidate_salary_recruiter_fee'
    elif bvh_id=='INSIGHTEXCHANGE': candidates=[max(a_product,b_product,a_case*.2,b_case*.2,25000*rate)]; method='advisory_or_insight_value_proxy'
    elif bvh_id=='TRUSTPATH': candidates=[max(a_product,b_product,a_case,b_case,a_pain,b_pain)*.25]; method='relationship_acceleration_value_proxy'
    elif bvh_id=='ECOSYSTEMAMPLIFICATION': candidates=[max(a_product,b_product,a_case,b_case)*.35]; method='ecosystem_reach_multiplier_proxy'
    else: candidates=[a_product,b_product,a_case,b_case,a_pain,b_pain]
    labelled=calibration.get('bvh_type_avg_value_gbp',{}).get(bvh_id,0)
    if labelled: candidates.append(float(labelled)*rate)
    potential=max([c for c in candidates if c is not None]+[0.0])
    if potential<=0: potential=50000*rate; method='fallback_minimum_value_proxy'
    probability=float(hvti_result.get('EstimatedProbabilityOfTransaction',0)); expected=potential*probability
    explainer=f"Estimated potential value uses {method}. Profile signals considered: PersonA product/value case ${max(a_product,a_case):,.0f}, PersonB product/value case ${max(b_product,b_case):,.0f}, buyer pain/cost signals ${max(a_pain,b_pain):,.0f}. Selected potential value is ${potential:,.0f}; expected value is ${expected:,.0f} after applying {probability:.1%} estimated transaction probability."
    return {'EstimatedHVITUSDValue':int(round(potential)),'EstimatedExpectedUSDValue':int(round(expected)),'EstimatedHVITUSDValue Explainer':explainer,'RoIModelMethod':method,'Currency':'USD','GBPToUSD':rate,'ValueSignals':{'PersonAProductOrValueCaseUSD':int(round(max(a_product,a_case))),'PersonBProductOrValueCaseUSD':int(round(max(b_product,b_case))),'PersonABuyerPainUSD':int(round(a_pain)),'PersonBBuyerPainUSD':int(round(b_pain))}}
