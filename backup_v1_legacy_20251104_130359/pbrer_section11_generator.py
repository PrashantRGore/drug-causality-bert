"""
PBRER Section 11 - Company Comment Generator
Generates drug-specific summaries with company interpretation
Following ICH E2C(R2) Section 11 format
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))

import re
from collections import defaultdict
from datetime import datetime
from inference import CausalityClassifier, extract_text_from_pdf, safe_sent_tokenize

# Import PBRER templates
from pbrer_psur_templates import (
    PBRER_LANGUAGE, PSUR_LANGUAGE, get_causality_statement
)

# Configuration
MODEL_PATH = './models/production_model_final'
THRESHOLD = 0.5


class PBRERSection11Generator:
    """
    Generates PBRER Section 11 compliant company comments
    for drug-specific causality assessment
    """
    
    def __init__(self, pdf_path, drug_name):
        self.pdf_path = pdf_path
        self.drug_name = drug_name
        self.classifier = CausalityClassifier(model_path=MODEL_PATH, threshold=THRESHOLD)
        self.pdf_text = None
        self.sentences = []
        self.drug_sentences = []
        self.analysis_results = {}
        
    def extract_and_analyze(self):
        """Extract text and analyze for drug-specific information"""
        print(f"\n{'='*80}")
        print(f"PBRER Section 11 Analysis - {self.drug_name}")
        print('='*80)
        
        # Extract text
        print("\n[1/4] Extracting text from PDF...")
        self.pdf_text = extract_text_from_pdf(self.pdf_path)
        print(f"✓ Extracted {len(self.pdf_text):,} characters")
        
        # Tokenize
        print("\n[2/4] Tokenizing sentences...")
        self.sentences = safe_sent_tokenize(self.pdf_text)
        print(f"✓ Found {len(self.sentences)} sentences")
        
        # Find drug-specific sentences
        print(f"\n[3/4] Identifying sentences mentioning '{self.drug_name}'...")
        drug_pattern = re.compile(rf'\b{re.escape(self.drug_name)}\b', re.IGNORECASE)
        
        for i, sent in enumerate(self.sentences, 1):
            if drug_pattern.search(sent):
                # Get causality prediction
                result = self.classifier.predict(sent, return_probs=True)
                
                self.drug_sentences.append({
                    'sentence_number': i,
                    'sentence': sent,
                    'prediction': result['prediction'],
                    'label': result['label'],
                    'confidence': result['confidence'],
                    'probability_related': result['probabilities']['related'],
                    'probability_not_related': result['probabilities']['not_related']
                })
        
        print(f"✓ Found {len(self.drug_sentences)} sentences mentioning '{self.drug_name}'")
        
        # Analyze content
        print(f"\n[4/4] Analyzing contextual information...")
        self._analyze_contextual_factors()
        print("✓ Analysis complete")
        
        return self.drug_sentences
    
    def _analyze_contextual_factors(self):
        """Extract contextual factors from drug-specific sentences"""
        
        # Initialize categories
        self.analysis_results = {
            'concomitant_medications': [],
            'time_to_onset': [],
            'concurrent_conditions': [],
            'confounding_factors': [],
            'dose_information': [],
            'dechallenge_rechallenge': [],
            'patient_demographics': [],
            'clinical_outcomes': [],
            'mechanism_information': [],
            'other_relevant_info': []
        }
        
        # Patterns for extraction
        patterns = {
            'concomitant_medications': [
                r'(?:concomitant|concurrent|co-administered|combination with|along with|together with)\s+(?:medication|drug|therapy|treatment)',
                r'(?:receiving|taking|administered)\s+(?:with|and)\s+([A-Z][a-z]+(?:mab|nib|prazole|statin|mycin|cillin))',
                r'(?:plus|combined with|in combination with)\s+([A-Z][a-z]+)'
            ],
            'time_to_onset': [
                r'(?:after|within|following)\s+(\d+\s*(?:day|week|month|year|hour)s?)',
                r'(?:latency|onset)\s+(?:period|time)?\s*(?:of|was)?\s*(\d+\s*(?:day|week|month|year)s?)',
                r'(\d+\s*(?:day|week|month|year)s?)\s+(?:after|following|post)\s+(?:initiation|administration|treatment)'
            ],
            'concurrent_conditions': [
                r'(?:underlying|pre-existing|concurrent|comorbid)\s+(?:condition|disease|disorder)',
                r'(?:history of|diagnosed with|suffering from)\s+([a-z\s]+(?:disease|disorder|syndrome|condition))',
                r'(?:patient|subject)\s+(?:with|had)\s+([a-z\s]+(?:disease|disorder|syndrome))'
            ],
            'dose_information': [
                r'(\d+\s*(?:mg|g|mcg|µg|IU|units?)(?:/(?:day|kg|m2))?)',
                r'(?:dose|dosage|administered)\s+(?:of|was)?\s*(\d+\s*(?:mg|g|mcg))',
                r'(?:increased|decreased|adjusted)\s+(?:to|from)\s+(\d+\s*(?:mg|g))'
            ],
            'dechallenge_rechallenge': [
                r'(?:discontinu|withdraw|stopp|ceas)(?:ed|ing|ation)',
                r'(?:rechallenge|reintroduc|readministr)(?:ed|ing|tion)',
                r'(?:upon|after)\s+(?:discontinuation|withdrawal|cessation)'
            ],
            'patient_demographics': [
                r'(\d+)[\s-]year[\s-]old',
                r'(?:male|female|man|woman)',
                r'(?:age|aged)\s+(\d+)',
                r'(?:elderly|pediatric|geriatric|adult)'
            ],
            'clinical_outcomes': [
                r'(?:recovered|resolved|improved|worsened|fatal|death)',
                r'(?:hospitalization|hospital admission|emergency)',
                r'(?:outcome|result|resolution)\s+(?:was|included)'
            ],
            'mechanism_information': [
                r'(?:mechanism|pathophysiology|etiology)',
                r'(?:due to|caused by|attributed to|related to)',
                r'(?:pharmacological|biological|physiological)\s+(?:mechanism|basis|explanation)'
            ]
        }
        
        # Extract information
        all_text = ' '.join([s['sentence'] for s in self.drug_sentences])
        
        for category, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.finditer(pattern, all_text, re.IGNORECASE)
                for match in matches:
                    context = all_text[max(0, match.start()-100):min(len(all_text), match.end()+100)]
                    self.analysis_results[category].append({
                        'match': match.group(0),
                        'context': context.strip()
                    })
        
        # Identify confounding factors
        confounding_keywords = [
            'confound', 'bias', 'alternative explanation', 'other cause',
            'may be due to', 'possibly related to', 'could be attributed to',
            'uncertain', 'unclear', 'unknown etiology'
        ]
        
        for sent_info in self.drug_sentences:
            sent_lower = sent_info['sentence'].lower()
            for keyword in confounding_keywords:
                if keyword in sent_lower:
                    self.analysis_results['confounding_factors'].append({
                        'sentence': sent_info['sentence'],
                        'keyword': keyword
                    })
                    break
    
    def generate_document_summary(self):
        """Generate comprehensive document summary focused on the drug"""
        
        summary = []
        
        # Title and source
        summary.append(f"**Document Summary for {self.drug_name}**\n")
        summary.append(f"Source: {Path(self.pdf_path).name}")
        summary.append(f"Analysis Date: {datetime.now().strftime('%d %B %Y')}\n")
        
        # Overview
        related_count = sum(1 for s in self.drug_sentences if s['label'] == 1)
        summary.append(f"**Overview:**")
        summary.append(f"This document contains {len(self.drug_sentences)} sentences mentioning {self.drug_name}, "
                      f"of which {related_count} demonstrate potential causality relationships with adverse events "
                      f"(confidence threshold: {THRESHOLD}).\n")
        
        # Key findings
        if related_count > 0:
            summary.append("**Key Causality Findings:**")
            for i, sent_info in enumerate([s for s in self.drug_sentences if s['label'] == 1], 1):
                summary.append(f"{i}. {sent_info['sentence'][:200]}{'...' if len(sent_info['sentence']) > 200 else ''}")
                summary.append(f"   (Confidence: {sent_info['probability_related']*100:.1f}%)\n")
        
        return '\n'.join(summary)
    
    def generate_contextual_information(self):
        """Generate detailed contextual information section"""
        
        info = []
        info.append(f"**Contextual Information for {self.drug_name}**\n")
        
        # Concomitant medications
        if self.analysis_results['concomitant_medications']:
            info.append("**Concomitant Medications:**")
            unique_meds = list({m['match'] for m in self.analysis_results['concomitant_medications']})
            for med in unique_meds[:5]:  # Top 5
                info.append(f"• {med}")
            info.append("")
        else:
            info.append("**Concomitant Medications:** Not specifically mentioned in the document.\n")
        
        # Time to onset
        if self.analysis_results['time_to_onset']:
            info.append("**Time to Onset:**")
            unique_times = list({t['match'] for t in self.analysis_results['time_to_onset']})
            for time in unique_times[:3]:
                info.append(f"• {time}")
            info.append("")
        else:
            info.append("**Time to Onset:** Not specified in the document.\n")
        
        # Concurrent conditions
        if self.analysis_results['concurrent_conditions']:
            info.append("**Concurrent Conditions/Comorbidities:**")
            unique_conditions = list({c['match'] for c in self.analysis_results['concurrent_conditions']})
            for condition in unique_conditions[:5]:
                info.append(f"• {condition}")
            info.append("")
        else:
            info.append("**Concurrent Conditions:** Not specifically mentioned.\n")
        
        # Dose information
        if self.analysis_results['dose_information']:
            info.append("**Dose Information:**")
            unique_doses = list({d['match'] for d in self.analysis_results['dose_information']})
            for dose in unique_doses[:5]:
                info.append(f"• {dose}")
            info.append("")
        
        # Dechallenge/Rechallenge
        if self.analysis_results['dechallenge_rechallenge']:
            info.append("**Dechallenge/Rechallenge Information:**")
            info.append(f"• Evidence of dechallenge/rechallenge mentioned in document")
            info.append("")
        
        # Patient demographics
        if self.analysis_results['patient_demographics']:
            info.append("**Patient Demographics:**")
            unique_demo = list({d['match'] for d in self.analysis_results['patient_demographics']})
            for demo in unique_demo[:5]:
                info.append(f"• {demo}")
            info.append("")
        
        # Clinical outcomes
        if self.analysis_results['clinical_outcomes']:
            info.append("**Clinical Outcomes:**")
            unique_outcomes = list({o['match'] for o in self.analysis_results['clinical_outcomes']})
            for outcome in unique_outcomes[:5]:
                info.append(f"• {outcome}")
            info.append("")
        
        return '\n'.join(info)
    
    def generate_confounding_factors(self):
        """Generate confounding factors section"""
        
        factors = []
        factors.append("**Confounding Factors and Alternative Explanations:**\n")
        
        if self.analysis_results['confounding_factors']:
            factors.append("The following potential confounding factors were identified:\n")
            for i, factor in enumerate(self.analysis_results['confounding_factors'][:5], 1):
                factors.append(f"{i}. {factor['sentence'][:200]}{'...' if len(factor['sentence']) > 200 else ''}\n")
        else:
            factors.append("No explicit confounding factors or alternative explanations were mentioned in the document.\n")
        
        # Additional considerations
        factors.append("**Additional Considerations:**")
        
        if not self.analysis_results['concomitant_medications']:
            factors.append("• Concomitant medication use not documented - potential confounding factor")
        
        if not self.analysis_results['concurrent_conditions']:
            factors.append("• Underlying medical conditions not specified - may influence causality assessment")
        
        if not self.analysis_results['time_to_onset']:
            factors.append("• Temporal relationship not clearly established")
        
        if not self.analysis_results['dechallenge_rechallenge']:
            factors.append("• No dechallenge/rechallenge data available")
        
        return '\n'.join(factors)
    
    def generate_company_comment(self):
        """Generate PBRER Section 11 compliant company comment"""
        
        # Calculate overall causality
        related_count = sum(1 for s in self.drug_sentences if s['label'] == 1)
        total_count = len(self.drug_sentences)
        
        if related_count == 0:
            overall_assessment = "not related"
            confidence_level = "low"
        elif related_count / total_count > 0.7:
            overall_assessment = "related"
            confidence_level = "high"
        elif related_count / total_count > 0.4:
            overall_assessment = "possibly related"
            confidence_level = "moderate"
        else:
            overall_assessment = "unlikely related"
            confidence_level = "low to moderate"
        
        # Get max confidence
        max_confidence = max([s['probability_related'] for s in self.drug_sentences]) if self.drug_sentences else 0
        
        comment = []
        comment.append("="*80)
        comment.append("COMPANY COMMENT - PBRER SECTION 11 FORMAT")
        comment.append("="*80)
        comment.append("")
        
        # Header
        comment.append(f"**Medicinal Product:** {self.drug_name}")
        comment.append(f"**Source Document:** {Path(self.pdf_path).name}")
        comment.append(f"**Assessment Date:** {datetime.now().strftime('%d %B %Y')}")
        comment.append(f"**Reporting Period:** As documented in source")
        comment.append("")
        
        # Summary of case/literature
        comment.append("**Summary of Literature/Case Report:**")
        comment.append("")
        
        # Brief summary
        if related_count > 0:
            comment.append(f"This {'literature article' if 'pdf' in self.pdf_path.lower() else 'case report'} "
                          f"describes {related_count} instance(s) of potential adverse events associated with "
                          f"{self.drug_name}. ")
            
            # Add key adverse events
            events = set()
            for sent in [s for s in self.drug_sentences if s['label'] == 1]:
                # Extract potential event terms
                event_patterns = [
                    r'\b(cataract|cataracts|neuropathy|neurotoxicity|hepatotoxicity|cardiotoxicity)\b',
                    r'\b(adverse event|adverse reaction|ADR|AE)\b'
                ]
                for pattern in event_patterns:
                    matches = re.findall(pattern, sent['sentence'], re.IGNORECASE)
                    events.update(matches)
            
            if events:
                comment.append(f"The reported adverse event(s) include: {', '.join(list(events)[:5])}. ")
            
            # Time to onset if available
            if self.analysis_results['time_to_onset']:
                time_info = self.analysis_results['time_to_onset'][0]['match']
                comment.append(f"Time to onset was reported as {time_info}. ")
            
            # Outcome if available
            if self.analysis_results['clinical_outcomes']:
                outcome_info = self.analysis_results['clinical_outcomes'][0]['match']
                comment.append(f"Clinical outcome: {outcome_info}. ")
        else:
            comment.append(f"This document mentions {self.drug_name} in {total_count} instance(s), "
                          f"however, no clear causality relationship with adverse events was identified "
                          f"based on the automated analysis.")
        
        comment.append("")
        
        # Company causality assessment
        comment.append("**Company Causality Assessment:**")
        comment.append("")
        
        if overall_assessment == "related":
            comment.append(f"Based on the available information and automated causality analysis "
                          f"(confidence: {max_confidence*100:.1f}%), a causal relationship between "
                          f"{self.drug_name} and the reported adverse event(s) is considered "
                          f"**{overall_assessment.upper()}**.")
            comment.append("")
            comment.append("**Rationale:**")
            comment.append(f"• Temporal association documented ({len([t for t in self.analysis_results['time_to_onset']])} reference(s))")
            comment.append(f"• Causality confidence score: {max_confidence*100:.1f}%")
            
            if self.analysis_results['dechallenge_rechallenge']:
                comment.append("• Dechallenge/rechallenge data available")
            
            if self.analysis_results['mechanism_information']:
                comment.append("• Biological plausibility supported by documented mechanism")
            
            comment.append(f"• Number of related statements: {related_count}/{total_count}")
            
        elif overall_assessment == "possibly related":
            comment.append(f"Based on the available information, a causal relationship between "
                          f"{self.drug_name} and the reported adverse event(s) is considered "
                          f"**{overall_assessment.upper()}**.")
            comment.append("")
            comment.append("**Rationale:**")
            comment.append(f"• Moderate evidence of causality (confidence: {max_confidence*100:.1f}%)")
            comment.append(f"• {related_count} of {total_count} statements suggest potential association")
            
            if self.analysis_results['confounding_factors']:
                comment.append(f"• {len(self.analysis_results['confounding_factors'])} potential confounding factor(s) identified")
            
            if not self.analysis_results['dechallenge_rechallenge']:
                comment.append("• Limited dechallenge/rechallenge information")
            
        else:
            comment.append(f"Based on the available information, a causal relationship between "
                          f"{self.drug_name} and adverse events is considered "
                          f"**{overall_assessment.upper()}**.")
            comment.append("")
            comment.append("**Rationale:**")
            comment.append(f"• Low causality confidence score: {max_confidence*100:.1f}%")
            comment.append(f"• Limited evidence in document ({related_count}/{total_count} statements)")
            
            if self.analysis_results['confounding_factors']:
                comment.append("• Alternative explanations identified")
        
        comment.append("")
        
        # Confounding factors
        comment.append("**Confounding Factors:**")
        comment.append("")
        
        confounders_listed = False
        
        if self.analysis_results['concomitant_medications']:
            comment.append(f"• Concomitant medications: {len(self.analysis_results['concomitant_medications'])} reference(s)")
            confounders_listed = True
        
        if self.analysis_results['concurrent_conditions']:
            comment.append(f"• Concurrent medical conditions: {len(self.analysis_results['concurrent_conditions'])} reference(s)")
            confounders_listed = True
        
        if self.analysis_results['confounding_factors']:
            comment.append(f"• Other confounding factors: {len(self.analysis_results['confounding_factors'])} identified")
            confounders_listed = True
        
        if not confounders_listed:
            comment.append("• No specific confounding factors explicitly mentioned in the document")
            comment.append("• However, absence of information does not exclude potential confounders")
        
        comment.append("")
        
        # Conclusion and recommendation
        comment.append("**Conclusion and Recommendation:**")
        comment.append("")
        
        if overall_assessment == "related":
            comment.append(f"This case/literature supports a causal association between {self.drug_name} "
                          f"and the reported adverse event(s). ")
            comment.append("")
            comment.append("**Recommended Actions:**")
            comment.append("• Include in cumulative review for next PSUR")
            comment.append("• Evaluate need for product information update")
            comment.append("• Consider for signal detection analysis")
            comment.append("• Continue pharmacovigilance monitoring")
            
        elif overall_assessment == "possibly related":
            comment.append(f"While a causal relationship cannot be definitively established, "
                          f"the possibility of an association between {self.drug_name} and the "
                          f"reported event(s) cannot be excluded. ")
            comment.append("")
            comment.append("**Recommended Actions:**")
            comment.append("• Include in routine pharmacovigilance monitoring")
            comment.append("• Document in periodic safety reports")
            comment.append("• Seek additional cases for cumulative review")
            comment.append("• No immediate regulatory action required")
            
        else:
            comment.append(f"Based on current evidence, a causal relationship between {self.drug_name} "
                          f"and adverse events is considered unlikely. However, continued monitoring "
                          f"is recommended as part of routine pharmacovigilance activities.")
            comment.append("")
            comment.append("**Recommended Actions:**")
            comment.append("• Continue routine pharmacovigilance")
            comment.append("• Document in periodic safety reports")
            comment.append("• No specific regulatory action required at this time")
        
        comment.append("")
        comment.append("="*80)
        comment.append(f"Assessment completed by: Drug Causality BERT v2.0")
        comment.append(f"Date: {datetime.now().strftime('%d %B %Y %H:%M:%S')}")
        comment.append("="*80)
        
        return '\n'.join(comment)
    
    def generate_complete_report(self):
        """Generate complete PBRER Section 11 report"""
        
        # Run analysis
        self.extract_and_analyze()
        
        # Generate all sections
        report = []
        
        report.append("\n" + "="*80)
        report.append(f"PBRER SECTION 11 - COMPLETE ANALYSIS")
        report.append(f"Medicinal Product: {self.drug_name}")
        report.append("="*80 + "\n")
        
        # Section 1: Document Summary
        report.append(self.generate_document_summary())
        report.append("\n" + "-"*80 + "\n")
        
        # Section 2: Contextual Information
        report.append(self.generate_contextual_information())
        report.append("\n" + "-"*80 + "\n")
        
        # Section 3: Confounding Factors
        report.append(self.generate_confounding_factors())
        report.append("\n" + "-"*80 + "\n")
        
        # Section 4: Company Comment
        report.append(self.generate_company_comment())
        
        return '\n'.join(report)


def generate_pbrer_section11(pdf_path, drug_name, output_path=None):
    """
    Main function to generate PBRER Section 11 report
    
    Args:
        pdf_path: Path to PDF document
        drug_name: Name of the drug to analyze
        output_path: Optional path to save report
    
    Returns:
        Complete report text
    """
    generator = PBRERSection11Generator(pdf_path, drug_name)
    report = generator.generate_complete_report()
    
    # Save if output path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n✓ Report saved to: {output_path}")
    
    return report, generator.analysis_results


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python pbrer_section11_generator.py <pdf_path> <drug_name>")
        print("Example: python pbrer_section11_generator.py article.pdf Adalimumab")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    drug_name = sys.argv[2]
    
    report, analysis = generate_pbrer_section11(pdf_path, drug_name)
    print(report)
