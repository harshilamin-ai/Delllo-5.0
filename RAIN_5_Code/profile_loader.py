from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
from .utils import read_csv, parse_json_cell, stable_id, token_set

JSON_FIELDS=['user_identity_json','previous_employment_and_education_history_json','offers_json','needs_json','insights_json','solutions_json','commercial_pricing_json','product_value_cases_json','use_cases_json','credibility_statements_json','buyer_pain_points_json','regulatory_drivers_json','commercial_context_json','relationship_edges_json','full_cil_profile_json']

def load_profiles(cil_profiles_path: Path) -> List[Dict[str, Any]]:
    rows=read_csv(cil_profiles_path); profiles=[]
    for idx,row in enumerate(rows,1):
        p=dict(row); name=p.get('full_name') or p.get('PersonName') or f'Profile {idx:04d}'
        p['PersonID']=p.get('PersonID') or stable_id('P', f'{idx}:{name}')
        p['ProfileIndex']=idx
        for field in JSON_FIELDS:
            p[field]=parse_json_cell(p.get(field), [] if field.endswith('_json') and field not in ['commercial_context_json','full_cil_profile_json'] else {})
        text_parts=[p.get(k,'') for k in ['full_name','current_role','organisation_name','organisation_archetype','seniority','role_family','persona_cluster','market_regime','primary_driver_description','secondary_driver_description']]
        text_parts += [str(p.get(k,'')) for k in ['offers_json','needs_json','solutions_json','insights_json','commercial_pricing_json','product_value_cases_json','use_cases_json','buyer_pain_points_json','regulatory_drivers_json']]
        p['_search_text']=' '.join(text_parts); p['_tokens']=token_set(p['_search_text'])
        profiles.append(p)
    return profiles
