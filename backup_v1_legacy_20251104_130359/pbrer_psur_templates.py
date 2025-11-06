"""
PBRER/PSUR Professional Medical Language Templates
Based on ICH E2C(R2) and regulatory authority guidelines
Version 2.0
"""

# PBRER (Periodic Benefit-Risk Evaluation Report) Language Patterns
PBRER_LANGUAGE = {
    'signal_detection': {
        'positive': [
            "A potential safety signal has been identified through disproportionality analysis",
            "Post-marketing surveillance data suggest a possible association",
            "Cumulative review of spontaneous reports indicates an emerging safety concern",
            "Signal detection algorithms have flagged this drug-event combination for further evaluation",
            "The observed reporting rate exceeds the expected background rate"
        ],
        'negative': [
            "No new safety signals were identified during the reporting period",
            "The benefit-risk profile remains favorable based on current evidence",
            "Routine pharmacovigilance activities have not revealed any unexpected safety concerns",
            "The observed event rate is consistent with the known safety profile"
        ]
    },
    'causality_assessment': {
        'definite': [
            "A definite causal relationship is established based on temporal association, positive dechallenge, and positive rechallenge",
            "The adverse event demonstrates a clear temporal relationship with drug administration, with resolution upon discontinuation and recurrence upon rechallenge",
            "Strong evidence supports a causal association, including biological plausibility and dose-response relationship"
        ],
        'probable': [
            "A probable causal relationship exists based on temporal association and absence of alternative explanations",
            "The adverse event is likely related to the medicinal product, considering the temporal sequence and pharmacological plausibility",
            "Available evidence suggests a probable association, though confounding factors cannot be entirely excluded"
        ],
        'possible': [
            "A possible causal relationship cannot be excluded based on temporal association",
            "The adverse event may be attributed to the medicinal product, although alternative etiologies are plausible",
            "Insufficient data preclude definitive causality assessment; however, a possible association warrants continued monitoring"
        ],
        'unlikely': [
            "The temporal relationship and clinical presentation suggest alternative etiologies are more likely",
            "Available evidence does not support a causal association with the medicinal product",
            "The adverse event is unlikely to be drug-related based on the known pharmacological profile"
        ]
    },
    'risk_characterization': {
        'serious': [
            "This adverse drug reaction (ADR) is classified as serious per ICH E2A criteria",
            "The event meets regulatory criteria for seriousness, requiring expedited reporting",
            "Medically significant adverse reaction warranting inclusion in core safety information"
        ],
        'important_identified': [
            "This represents an important identified risk requiring risk minimization measures",
            "The adverse reaction is documented in the Reference Safety Information (RSI)",
            "Established safety concern requiring ongoing monitoring and risk communication"
        ],
        'important_potential': [
            "This constitutes an important potential risk requiring further characterization",
            "Emerging safety signal warranting enhanced pharmacovigilance activities",
            "Potential risk under investigation through post-authorization safety studies"
        ]
    }
}

# PSUR (Periodic Safety Update Report) Language Patterns
PSUR_LANGUAGE = {
    'executive_summary': [
        "This Periodic Safety Update Report (PSUR) covers the international safety experience for {drug} during the reporting interval from {start_date} to {end_date}",
        "The cumulative safety database comprises {total_cases} cases, including {serious_cases} serious adverse events",
        "No clinically significant changes to the benefit-risk profile were identified during this reporting period",
        "The overall safety profile remains consistent with the current Reference Safety Information"
    ],
    'signal_evaluation': [
        "Systematic signal detection was performed using disproportionality analysis (ROR, PRR, IC)",
        "Signals meeting predefined thresholds underwent detailed medical review and causality assessment",
        "The signal evaluation process adhered to Good Pharmacovigilance Practices (GVP) Module IX",
        "Validated signals were assessed for clinical significance and need for regulatory action"
    ],
    'benefit_risk_assessment': [
        "The benefit-risk balance remains positive for approved indications",
        "No new safety information has emerged that would alter the current benefit-risk assessment",
        "The therapeutic benefit continues to outweigh identified and potential risks when used according to approved prescribing information",
        "Risk minimization measures are adequate and effective based on available evidence"
    ],
    'conclusion': [
        "Based on cumulative review of safety data, no changes to the product information are warranted at this time",
        "Continued routine pharmacovigilance is recommended to monitor the evolving safety profile",
        "The sponsor will continue to monitor this safety concern through ongoing surveillance activities",
        "No additional risk minimization activities are required beyond those currently implemented"
    ]
}

# Medical Terminology Standards
MEDICAL_TERMINOLOGY = {
    'adverse_events': {
        'severity': {
            'mild': 'Grade 1 (mild): Asymptomatic or mild symptoms; clinical or diagnostic observations only; intervention not indicated',
            'moderate': 'Grade 2 (moderate): Minimal, local, or noninvasive intervention indicated; limiting age-appropriate instrumental ADL',
            'severe': 'Grade 3 (severe): Medically significant but not immediately life-threatening; hospitalization or prolongation of hospitalization indicated',
            'life_threatening': 'Grade 4 (life-threatening): Urgent intervention indicated',
            'fatal': 'Grade 5 (death): Death related to adverse event'
        },
        'seriousness_criteria': [
            'Results in death',
            'Is life-threatening',
            'Requires inpatient hospitalization or prolongation of existing hospitalization',
            'Results in persistent or significant disability/incapacity',
            'Is a congenital anomaly/birth defect',
            'Is a medically important event'
        ],
        'outcome': [
            'Recovered/Resolved',
            'Recovering/Resolving',
            'Not Recovered/Not Resolved',
            'Recovered/Resolved with Sequelae',
            'Fatal',
            'Unknown'
        ]
    },
    'causality_terms': {
        'temporal': [
            'temporal association',
            'latency period',
            'time to onset',
            'duration of exposure',
            'dechallenge',
            'rechallenge'
        ],
        'biological': [
            'pharmacological plausibility',
            'mechanism of action',
            'dose-response relationship',
            'biological gradient',
            'pathophysiological mechanism'
        ],
        'epidemiological': [
            'incidence rate',
            'prevalence',
            'reporting rate',
            'background rate',
            'relative risk',
            'odds ratio',
            'confidence interval'
        ]
    }
}

# Regulatory Action Language
REGULATORY_ACTIONS = {
    'label_updates': {
        'warnings': [
            "Addition to Warnings and Precautions section is recommended",
            "Enhanced warning language is warranted based on post-marketing experience",
            "Inclusion in Boxed Warning requires consideration given the severity and preventability"
        ],
        'adverse_reactions': [
            "Addition to Adverse Reactions section with frequency estimate",
            "Update to reflect post-marketing surveillance findings",
            "Inclusion of event with appropriate frequency category (common, uncommon, rare, very rare)"
        ],
        'contraindications': [
            "Contraindication is recommended for patients with specific risk factors",
            "Absolute contraindication warranted based on serious safety concerns",
            "Relative contraindication with appropriate risk-benefit consideration"
        ]
    },
    'risk_minimization': {
        'routine': [
            "Current product labeling provides adequate risk information",
            "Standard pharmacovigilance activities are sufficient",
            "No additional risk minimization measures required"
        ],
        'additional': [
            "Healthcare Professional Communication (DHPC) is recommended",
            "Patient information materials should be developed and distributed",
            "Risk Evaluation and Mitigation Strategy (REMS) may be warranted",
            "Controlled distribution program should be considered",
            "Enhanced pharmacovigilance monitoring is recommended"
        ]
    },
    'further_investigation': [
        "Post-authorization safety study (PASS) is recommended to further characterize the risk",
        "Registry-based surveillance should be implemented to monitor long-term outcomes",
        "Pharmacoepidemiological study is warranted to quantify the risk",
        "Clinical trial data should be re-analyzed with focus on this safety outcome",
        "Literature review and meta-analysis should be conducted"
    ]
}

# Statistical and Epidemiological Language
STATISTICAL_LANGUAGE = {
    'disproportionality': {
        'ror': "Reporting Odds Ratio (ROR) = {value} [95% CI: {lower}-{upper}]",
        'prr': "Proportional Reporting Ratio (PRR) = {value} [95% CI: {lower}-{upper}]",
        'ic': "Information Component (IC) = {value} [95% CI: {lower}-{upper}]",
        'ebgm': "Empirical Bayes Geometric Mean (EBGM) = {value} [95% CI: {lower}-{upper}]"
    },
    'interpretation': {
        'significant': "The disproportionality analysis demonstrates a statistically significant signal (p < 0.05)",
        'non_significant': "No statistically significant disproportionality was observed",
        'threshold_met': "The signal meets predefined thresholds for further evaluation",
        'threshold_not_met': "The signal does not meet criteria for regulatory action"
    },
    'frequency_categories': {
        'very_common': "Very common (≥1/10)",
        'common': "Common (≥1/100 to <1/10)",
        'uncommon': "Uncommon (≥1/1,000 to <1/100)",
        'rare': "Rare (≥1/10,000 to <1/1,000)",
        'very_rare': "Very rare (<1/10,000)",
        'not_known': "Not known (cannot be estimated from available data)"
    }
}

# Clinical Significance Assessment
CLINICAL_SIGNIFICANCE = {
    'high': [
        "This adverse event is of high clinical significance due to potential for serious morbidity or mortality",
        "The event represents a clinically important safety concern requiring immediate attention",
        "Given the severity and potential preventability, this warrants priority risk minimization efforts"
    ],
    'moderate': [
        "This adverse event is of moderate clinical significance, requiring appropriate monitoring and management",
        "The event may impact patient quality of life and requires healthcare provider awareness",
        "Clinical significance is moderate; routine monitoring and patient counseling are recommended"
    ],
    'low': [
        "This adverse event is of low clinical significance with minimal impact on patient outcomes",
        "The event is generally self-limiting and does not require specific intervention",
        "Clinical significance is low; however, documentation and monitoring remain important"
    ]
}

# Recommendations Template
RECOMMENDATIONS_TEMPLATE = {
    'monitoring': [
        "Enhanced pharmacovigilance monitoring is recommended for the next reporting period",
        "Targeted follow-up of reported cases should be conducted to obtain additional clinical details",
        "Active surveillance in high-risk populations is warranted",
        "Real-world evidence studies should be considered to better characterize the risk"
    ],
    'communication': [
        "Healthcare provider communication should be issued to raise awareness",
        "Patient information materials should be updated to reflect current safety information",
        "Regulatory authorities should be notified of this emerging safety signal",
        "Scientific advisory committee consultation may be beneficial"
    ],
    'labeling': [
        "Product information should be updated to include this adverse reaction",
        "Risk communication in the Summary of Product Characteristics (SmPC) should be enhanced",
        "Patient information leaflet should be revised to include appropriate warnings",
        "Prescribing information should reflect current understanding of the risk"
    ]
}

def get_pbrer_language(category, subcategory=None):
    """Get PBRER-appropriate language"""
    if subcategory:
        return PBRER_LANGUAGE.get(category, {}).get(subcategory, [])
    return PBRER_LANGUAGE.get(category, [])

def get_psur_language(category):
    """Get PSUR-appropriate language"""
    return PSUR_LANGUAGE.get(category, [])

def get_causality_statement(who_category, naranjo_category, confidence):
    """Generate professional causality statement"""
    if confidence > 0.95 and who_category in ['Certain/Definite', 'Probable/Likely']:
        return PBRER_LANGUAGE['causality_assessment']['definite'][0]
    elif confidence > 0.80 and who_category == 'Probable/Likely':
        return PBRER_LANGUAGE['causality_assessment']['probable'][0]
    elif confidence > 0.60:
        return PBRER_LANGUAGE['causality_assessment']['possible'][0]
    else:
        return PBRER_LANGUAGE['causality_assessment']['unlikely'][0]

def get_regulatory_recommendation(classification, confidence, event_severity='moderate'):
    """Generate regulatory recommendation"""
    if classification == 'related' and confidence > 0.95:
        if event_severity == 'serious':
            return REGULATORY_ACTIONS['label_updates']['warnings'][0]
        else:
            return REGULATORY_ACTIONS['label_updates']['adverse_reactions'][0]
    elif classification == 'related' and confidence > 0.80:
        return REGULATORY_ACTIONS['risk_minimization']['additional'][0]
    else:
        return REGULATORY_ACTIONS['risk_minimization']['routine'][0]
