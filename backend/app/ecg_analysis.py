"""
ECG Analysis module for running models and parsing results.

This module provides functionality to:
- Run selected ECG analysis models via Docker
- Parse probability CSV outputs
- Classify results with risk thresholds
- Categorize 77-class diagnoses
- Convert 8-lead ECG files to 12-lead format
"""
import asyncio
import base64
import csv
import logging
import os
import shutil
import struct
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal, Tuple

from .docker_control import _run_command, get_docker_status, CONTAINER_NAME
from .settings import settings

logger = logging.getLogger(__name__)


def detect_and_convert_encoding(file_path: str) -> Tuple[bool, str, str]:
    """
    Detect file encoding and convert UTF-16 to UTF-8 if necessary.

    Many Philips ECG devices export XML files in UTF-16 encoding, which
    HeartWise doesn't handle properly. This function detects the encoding
    and converts to UTF-8 if needed.

    Args:
        file_path: Path to the XML file

    Returns:
        Tuple of (was_converted, original_encoding, message)
    """
    try:
        with open(file_path, 'rb') as f:
            raw_bytes = f.read(4)

        # Detect encoding from BOM (Byte Order Mark)
        original_encoding = None

        # UTF-16 LE BOM: FF FE
        if raw_bytes[:2] == b'\xff\xfe':
            original_encoding = 'utf-16-le'
        # UTF-16 BE BOM: FE FF
        elif raw_bytes[:2] == b'\xfe\xff':
            original_encoding = 'utf-16-be'
        # UTF-8 BOM: EF BB BF
        elif raw_bytes[:3] == b'\xef\xbb\xbf':
            original_encoding = 'utf-8-sig'
        # Check for UTF-16 without BOM (spaces between characters indicate UTF-16)
        elif len(raw_bytes) >= 4:
            # If second byte is 0x00, likely UTF-16 LE without BOM
            if raw_bytes[1] == 0x00 and raw_bytes[0] != 0x00:
                original_encoding = 'utf-16-le'
            # If first byte is 0x00, likely UTF-16 BE without BOM
            elif raw_bytes[0] == 0x00 and raw_bytes[1] != 0x00:
                original_encoding = 'utf-16-be'

        # If UTF-16 detected, convert to UTF-8
        if original_encoding and 'utf-16' in original_encoding:
            logger.info(f"Detected {original_encoding} encoding in {file_path}, converting to UTF-8...")

            # Read file with detected encoding
            with open(file_path, 'r', encoding=original_encoding) as f:
                content = f.read()

            # Remove BOM if present at start of content
            if content and content[0] == '\ufeff':
                content = content[1:]

            # Write back as UTF-8
            with open(file_path, 'w', encoding='utf-8') as f:
                # Update the XML declaration to specify UTF-8
                if content.startswith('<?xml'):
                    # Replace encoding declaration
                    import re
                    content = re.sub(
                        r'encoding\s*=\s*["\'][^"\']*["\']',
                        'encoding="UTF-8"',
                        content,
                        count=1
                    )
                f.write(content)

            logger.info(f"Successfully converted {file_path} from {original_encoding} to UTF-8")
            return True, original_encoding, f"Converted from {original_encoding} to UTF-8"

        return False, original_encoding or 'utf-8', "No conversion needed"

    except Exception as e:
        logger.error(f"Error detecting/converting encoding for {file_path}: {e}")
        return False, 'unknown', f"Error: {str(e)}"


def expand_8_to_12_leads(xml_path: str) -> Tuple[bool, str]:
    """
    Expand an 8-lead ECG XML file to 12-lead format by calculating derived leads.

    Standard 12-lead ECG has: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
    8-lead files typically have: I, II, V1, V2, V3, V4, V5, V6

    Derived leads are calculated as:
    - III = II - I
    - aVR = -(I + II) / 2
    - aVL = I - II/2
    - aVF = II - I/2

    IMPORTANT: GE MUSE XML files have multiple Waveform sections (Median ~600 samples,
    Rhythm ~5000 samples). This function processes EACH section independently,
    extracting leads from that section and calculating derived leads with the
    correct sample count for that section.

    Returns:
        Tuple of (success, message_or_path)
    """
    import re

    def extract_leads_from_section(section_content: str) -> Dict[str, Tuple[List[int], float]]:
        """Extract leads from a single Waveform section. Returns dict of lead_id -> (samples, amplitude_units)."""
        leads = {}

        # Find all LeadData elements in this section
        lead_pattern = re.compile(r'<LeadData>(.*?)</LeadData>', re.DOTALL | re.IGNORECASE)

        for lead_match in lead_pattern.finditer(section_content):
            lead_content = lead_match.group(1)

            # Extract LeadID
            lead_id_match = re.search(r'<LeadID>([^<]+)</LeadID>', lead_content, re.IGNORECASE)
            if not lead_id_match:
                continue
            lead_id = lead_id_match.group(1).strip()

            # Extract WaveFormData
            waveform_match = re.search(r'<WaveFormData>([^<]+)</WaveFormData>', lead_content, re.IGNORECASE)
            if not waveform_match:
                continue
            waveform_data = waveform_match.group(1)

            # Extract amplitude units
            amplitude_units = 1.0
            amp_match = re.search(r'<LeadAmplitudeUnitsPerBit>([^<]+)</LeadAmplitudeUnitsPerBit>', lead_content, re.IGNORECASE)
            if amp_match:
                try:
                    amplitude_units = float(amp_match.group(1))
                except:
                    pass

            # Decode base64 waveform data
            waveform_data_clean = ''.join(waveform_data.split())
            try:
                raw_bytes = base64.b64decode(waveform_data_clean)
                num_samples = len(raw_bytes) // 2
                samples = list(struct.unpack(f'<{num_samples}h', raw_bytes))
                leads[lead_id] = (samples, amplitude_units)
            except Exception as e:
                logger.warning(f"Failed to decode lead {lead_id}: {e}")

        return leads

    def calculate_derived_leads(lead_i: List[int], lead_ii: List[int]) -> Dict[str, List[int]]:
        """Calculate derived leads III, aVR, aVL, aVF from leads I and II."""
        n = len(lead_i)
        return {
            'III': [lead_ii[i] - lead_i[i] for i in range(n)],
            'aVR': [int(-(lead_i[i] + lead_ii[i]) / 2) for i in range(n)],
            'aVL': [int(lead_i[i] - lead_ii[i] / 2) for i in range(n)],
            'aVF': [int(lead_ii[i] - lead_i[i] / 2) for i in range(n)],
        }

    def build_lead_xml(lead_id: str, samples: List[int], amplitude_units: float, indent: str) -> str:
        """Build XML string for a LeadData element."""
        raw_bytes = struct.pack(f'<{len(samples)}h', *samples)
        b64_data = base64.b64encode(raw_bytes).decode('ascii')

        return f'''
{indent}<LeadData>
{indent}   <LeadByteCountTotal>{len(raw_bytes)}</LeadByteCountTotal>
{indent}   <LeadTimeOffset>0</LeadTimeOffset>
{indent}   <LeadSampleCountTotal>{len(samples)}</LeadSampleCountTotal>
{indent}   <LeadAmplitudeUnitsPerBit>{amplitude_units}</LeadAmplitudeUnitsPerBit>
{indent}   <LeadAmplitudeUnits>MICROVOLTS</LeadAmplitudeUnits>
{indent}   <LeadHighLimit>32767</LeadHighLimit>
{indent}   <LeadLowLimit>-32768</LeadLowLimit>
{indent}   <LeadID>{lead_id}</LeadID>
{indent}   <LeadOffsetFirstSample>0</LeadOffsetFirstSample>
{indent}   <FirstSampleBaseline>0</FirstSampleBaseline>
{indent}   <LeadSampleSize>2</LeadSampleSize>
{indent}   <LeadOff>FALSE</LeadOff>
{indent}   <LeadOn>TRUE</LeadOn>
{indent}   <WaveFormData>{b64_data}</WaveFormData>
{indent}</LeadData>'''

    try:
        # Read the original file preserving encoding
        with open(xml_path, 'rb') as f:
            original_bytes = f.read()

        # Try to detect encoding from XML declaration
        encoding = 'utf-8'
        encoding_match = re.search(rb'encoding=["\']([^"\']+)["\']', original_bytes[:200])
        if encoding_match:
            encoding = encoding_match.group(1).decode('ascii')

        original_content = original_bytes.decode(encoding)

        # Find ALL Waveform sections (Median, Rhythm, etc.)
        waveform_pattern = re.compile(r'(<Waveform[^>]*>)(.*?)(</Waveform>)', re.DOTALL | re.IGNORECASE)
        waveform_matches = list(waveform_pattern.finditer(original_content))

        if not waveform_matches:
            logger.info("No Waveform sections found in XML")
            return True, xml_path

        logger.info(f"Found {len(waveform_matches)} Waveform section(s)")

        # Detect indentation from original file
        indent_match = re.search(r'\n(\s+)<LeadData>', original_content)
        indent = indent_match.group(1) if indent_match else '      '

        # Check the first section to see if we need to expand
        first_section_leads = extract_leads_from_section(waveform_matches[0].group(2))
        existing_lead_ids = set(first_section_leads.keys())

        # Check for existing derived leads (case-insensitive)
        existing_upper = {lid.upper() for lid in existing_lead_ids}
        has_iii = 'III' in existing_upper
        has_avr = 'AVR' in existing_upper
        has_avl = 'AVL' in existing_upper
        has_avf = 'AVF' in existing_upper

        if has_iii and has_avr and has_avl and has_avf:
            logger.info("All 12 leads already present - no expansion needed")
            return True, xml_path

        if 'I' not in existing_lead_ids or 'II' not in existing_lead_ids:
            logger.info("No I and II leads found - cannot calculate derived leads")
            return True, xml_path

        logger.info(f"Found leads in first section: {list(existing_lead_ids)}")
        logger.info(f"Missing leads - III: {not has_iii}, aVR: {not has_avr}, aVL: {not has_avl}, aVF: {not has_avf}")

        # Process each Waveform section from end to beginning (to preserve positions)
        new_content = original_content
        sections_processed = 0

        for match in reversed(waveform_matches):
            section_start = match.start()
            section_end = match.end()
            section_open_tag = match.group(1)
            section_content = match.group(2)
            section_close_tag = match.group(3)

            # Extract leads from THIS section
            section_leads = extract_leads_from_section(section_content)

            if 'I' not in section_leads or 'II' not in section_leads:
                logger.warning(f"Section missing I or II leads, skipping")
                continue

            lead_i_data, amp_units = section_leads['I']
            lead_ii_data, _ = section_leads['II']

            # Verify same sample count
            if len(lead_i_data) != len(lead_ii_data):
                logger.warning(f"Lead I ({len(lead_i_data)} samples) and II ({len(lead_ii_data)} samples) have different lengths")
                continue

            logger.info(f"Processing section with {len(lead_i_data)} samples per lead")

            # Calculate derived leads for THIS section
            derived = calculate_derived_leads(lead_i_data, lead_ii_data)

            # Build XML for leads that are missing
            new_lead_elements = []
            if not has_iii:
                new_lead_elements.append(build_lead_xml('III', derived['III'], amp_units, indent))
            if not has_avr:
                new_lead_elements.append(build_lead_xml('aVR', derived['aVR'], amp_units, indent))
            if not has_avl:
                new_lead_elements.append(build_lead_xml('aVL', derived['aVL'], amp_units, indent))
            if not has_avf:
                new_lead_elements.append(build_lead_xml('aVF', derived['aVF'], amp_units, indent))

            if new_lead_elements:
                # Find the last </LeadData> in this section to insert after it
                last_lead_end = section_content.rfind('</LeadData>')
                if last_lead_end != -1:
                    last_lead_end += len('</LeadData>')

                    # Insert new leads after the last LeadData
                    new_section_content = (
                        section_content[:last_lead_end] +
                        ''.join(new_lead_elements) +
                        section_content[last_lead_end:]
                    )

                    # Replace the section in the full content
                    new_section = section_open_tag + new_section_content + section_close_tag
                    new_content = new_content[:section_start] + new_section + new_content[section_end:]

                    sections_processed += 1
                    logger.info(f"Inserted {len(new_lead_elements)} derived leads ({len(lead_i_data)} samples each)")

        if sections_processed == 0:
            logger.warning("No sections were expanded")
            return True, xml_path

        # Update NumberofLeads if present
        new_content = re.sub(
            r'<NumberofLeads>\d+</NumberofLeads>',
            '<NumberofLeads>12</NumberofLeads>',
            new_content
        )

        # Save the modified XML preserving original encoding
        output_path = xml_path.replace('.xml', '_12lead.xml')
        with open(output_path, 'wb') as f:
            f.write(new_content.encode(encoding))

        logger.info(f"Saved 12-lead XML to: {output_path} (expanded {sections_processed} sections)")
        return True, output_path

    except ET.ParseError as e:
        logger.error(f"XML parse error: {e}")
        return False, f"XML parse error: {e}"
    except Exception as e:
        logger.error(f"Error expanding leads: {e}", exc_info=True)
        return False, str(e)

# Available models configuration
AVAILABLE_MODELS = {
    # EfficientNet models
    "efficientnet_77": {
        "name": "77 Classes ECG (EfficientNet)",
        "docker_diagnosis": "ecg_machine_diagnosis",
        "docker_file_col": "77_classes_ecg_file_name",
        "architecture": "efficientnet",
        "type": "multi_label",
        "use_efficientnet": True,
        "use_wcr": False,
    },
    "efficientnet_lvef_40": {
        "name": "LVEF ≤40% (EfficientNet)",
        "docker_diagnosis": "lvef_40",
        "docker_file_col": "lvef_40_ecg_file_name",
        "architecture": "efficientnet",
        "type": "binary",
        "use_efficientnet": True,
        "use_wcr": False,
    },
    "efficientnet_lvef_50": {
        "name": "LVEF <50% (EfficientNet)",
        "docker_diagnosis": "lvef_50",
        "docker_file_col": "lvef_50_ecg_file_name",
        "architecture": "efficientnet",
        "type": "binary",
        "use_efficientnet": True,
        "use_wcr": False,
    },
    "efficientnet_afib_5y": {
        "name": "Risque FA 5 ans (EfficientNet)",
        "docker_diagnosis": "afib_5y",
        "docker_file_col": "afib_ecg_file_name",
        "architecture": "efficientnet",
        "type": "binary",
        "use_efficientnet": True,
        "use_wcr": False,
    },
    # WCR Transformer models
    # Note: WCR binary models (LVEF, AF) are NOT available in HeartWise Docker
    # despite being mentioned on GitHub - only 77 classes WCR is implemented
    "wcr_77": {
        "name": "77 Classes ECG (WCR Transformer)",
        "docker_diagnosis": "ecg_machine_diagnosis",
        "docker_file_col": "77_classes_ecg_file_name",
        "architecture": "wcr",
        "type": "multi_label",
        "use_efficientnet": False,
        "use_wcr": True,
    },
}

# ECG Categories from Docker constants
ECG_CATEGORIES = {
    "Rhythm Disorders": [
        "Ventricular tachycardia", "Bradycardia", "Brugada",
        "Wolff-Parkinson-White (Pre-excitation syndrome)", "Atrial flutter",
        "Ectopic atrial rhythm (< 100 BPM)", "Atrial tachycardia (>= 100 BPM)",
        "Sinusal", "Ventricular Rhythm", "Supraventricular tachycardia",
        "Junctional rhythm", "Regular", "Regularly irregular",
        "Irregularly irregular", "Afib", "Premature ventricular complex",
        "Premature atrial complex"
    ],
    "Conduction Disorder": [
        "Left anterior fascicular block", "Delta wave",
        "2nd degree AV block - mobitz 2", "Left bundle branch block",
        "Right bundle branch block", "Left axis deviation", "Atrial paced",
        "Right axis deviation", "Left posterior fascicular block",
        "1st degree AV block", "Right superior axis",
        "Nonspecific intraventricular conduction delay", "Third Degree AV Block",
        "2nd degree AV block - mobitz 1", "Prolonged QT", "U wave",
        "LV pacing", "Ventricular paced"
    ],
    "Enlargement of the heart chambers": [
        "Bi-atrial enlargement", "Left atrial enlargement",
        "Right atrial enlargement", "Left ventricular hypertrophy",
        "Right ventricular hypertrophy"
    ],
    "Pericarditis": ["Acute pericarditis"],
    "Infarction or ischemia": [
        "Q wave (septal- V1-V2)", "ST elevation (anterior - V3-V4)",
        "Q wave (posterior - V7-V9)", "Q wave (inferior - II, III, aVF)",
        "Q wave (anterior - V3-V4)", "ST elevation (lateral - I, aVL, V5-V6)",
        "Q wave (lateral- I, aVL, V5-V6)",
        "ST depression (lateral - I, avL, V5-V6)", "Acute MI",
        "ST elevation (septal - V1-V2)", "ST elevation (inferior - II, III, aVF)",
        "ST elevation (posterior - V7-V8-V9)",
        "ST depression (inferior - II, III, aVF)",
        "ST depression (anterior - V3-V4)"
    ],
    "Other diagnoses": [
        "ST downslopping", "ST depression (septal- V1-V2)",
        "R/S ratio in V1-V2 >1", "RV1 + SV6 > 11 mm", "Polymorph",
        "rSR' in V1-V2", "QRS complex negative in III", "qRS in V5-V6-I, aVL",
        "QS complex in V1-V2-V3", "R complex in V5-V6", "RaVL > 11 mm",
        "T wave inversion (septal- V1-V2)", "SV1 + RV5 or RV6 > 35 mm",
        "T wave inversion (inferior - II, III, aVF)", "Monomorph",
        "T wave inversion (anterior - V3-V4)",
        "T wave inversion (lateral -I, aVL, V5-V6)", "Low voltage",
        "Lead misplacement", "Early repolarization", "ST upslopping", "no_qrs"
    ]
}

# Thresholds for 77-class diagnoses (from Docker BERT_THRESHOLDS)
DIAGNOSIS_THRESHOLDS = {
    "Sinusal": 0.43, "Regular": 0.48, "Monomorph": 0.51,
    "QS complex in V1-V2-V3": 0.57, "R complex in V5-V6": 0.4,
    "T wave inversion (inferior - II, III, aVF)": 0.6,
    "Left bundle branch block": 0.31, "RaVL > 11 mm": 0.65,
    "SV1 + RV5 or RV6 > 35 mm": 0.48,
    "T wave inversion (lateral -I, aVL, V5-V6)": 0.59,
    "T wave inversion (anterior - V3-V4)": 0.58, "Left axis deviation": 0.46,
    "Left ventricular hypertrophy": 0.38, "Bradycardia": 0.57,
    "Q wave (inferior - II, III, aVF)": 0.46, "Afib": 0.46,
    "Irregularly irregular": 0.58, "Atrial tachycardia (>= 100 BPM)": 0.39,
    "Nonspecific intraventricular conduction delay": 0.34,
    "Premature ventricular complex": 0.34, "Polymorph": 0.61,
    "T wave inversion (septal- V1-V2)": 0.65, "Right bundle branch block": 0.38,
    "Ventricular paced": 0.34, "ST elevation (anterior - V3-V4)": 0.46,
    "ST elevation (septal - V1-V2)": 0.48, "1st degree AV block": 0.31,
    "Premature atrial complex": 0.33, "Atrial flutter": 0.44,
    "rSR' in V1-V2": 0.56, "qRS in V5-V6-I, aVL": 0.63,
    "Left anterior fascicular block": 0.45, "Right axis deviation": 0.49,
    "2nd degree AV block - mobitz 1": 0.51,
    "ST depression (inferior - II, III, aVF)": 0.51, "Acute pericarditis": 0.38,
    "ST elevation (inferior - II, III, aVF)": 0.36, "Low voltage": 0.5,
    "Regularly irregular": 0.58, "Junctional rhythm": 0.43,
    "Left atrial enlargement": 0.52,
    "ST elevation (lateral - I, aVL, V5-V6)": 0.46, "Atrial paced": 0.42,
    "Right ventricular hypertrophy": 0.38, "Delta wave": 0.3,
    "Wolff-Parkinson-White (Pre-excitation syndrome)": 0.28,
    "Prolonged QT": 0.4, "ST depression (anterior - V3-V4)": 0.48,
    "QRS complex negative in III": 0.56,
    "Q wave (lateral- I, aVL, V5-V6)": 0.51,
    "Supraventricular tachycardia": 0.42, "ST downslopping": 0.37,
    "ST depression (lateral - I, avL, V5-V6)": 0.51,
    "2nd degree AV block - mobitz 2": 0.37, "U wave": 0.26,
    "R/S ratio in V1-V2 >1": 0.52, "RV1 + SV6 > 11 mm": 0.53,
    "Left posterior fascicular block": 0.35, "Right atrial enlargement": 0.26,
    "ST depression (septal- V1-V2)": 0.41, "Q wave (septal- V1-V2)": 0.51,
    "Q wave (anterior - V3-V4)": 0.37, "ST upslopping": 0.39,
    "Right superior axis": 0.43, "Ventricular tachycardia": 0.35,
    "ST elevation (posterior - V7-V8-V9)": 0.4,
    "Ectopic atrial rhythm (< 100 BPM)": 0.4, "Lead misplacement": 0.32,
    "Third Degree AV Block": 0.37, "Acute MI": 0.38,
    "Early repolarization": 0.4, "Q wave (posterior - V7-V9)": 0.34,
    "Bi-atrial enlargement": 0.29, "LV pacing": 0.28, "Brugada": 0.22,
    "Ventricular Rhythm": 0.33, "no_qrs": 0.27
}

# Binary model thresholds (default 0.5)
BINARY_THRESHOLDS = {
    "lvef_40": 0.5,
    "lvef_50": 0.5,
    "afib_5y": 0.5,
}

# "Normal" baseline findings - these are GOOD when detected (high probability)
# We don't want to flag these as "abnormal" when they have high probability
NORMAL_BASELINE_FINDINGS = {
    "Sinusal",          # Normal sinus rhythm
    "Regular",          # Regular rhythm
    "Monomorph",        # Monomorphic (uniform) complexes
    "QS complex in V1-V2-V3",  # Can be normal variant
    "R complex in V5-V6",      # Normal R wave progression
}


def classify_risk(
    probability: float,
    threshold: float,
    is_normal_finding: bool = False
) -> Literal["normal", "borderline", "abnormal"]:
    """
    Classify risk based on probability and threshold.

    For PATHOLOGICAL findings (is_normal_finding=False):
    - Normal (green): prob < threshold * 0.5
    - Borderline (yellow): threshold * 0.5 <= prob < threshold
    - Abnormal (red): prob >= threshold

    For NORMAL BASELINE findings (is_normal_finding=True):
    - Detection of normal findings is GOOD (not flagged as abnormal)
    - Normal (green): prob >= threshold (finding IS present - good)
    - Borderline (yellow): prob >= threshold * 0.5 (might be present)
    - Abnormal (red): prob < threshold * 0.5 (finding NOT present - concerning)
    """
    if is_normal_finding:
        # For normal findings, INVERT the logic
        # High probability = finding is present = good (normal)
        if probability >= threshold:
            return "normal"
        elif probability >= threshold * 0.5:
            return "borderline"
        return "abnormal"  # Absence of normal finding is concerning
    else:
        # For pathological findings, standard logic
        if probability >= threshold:
            return "abnormal"
        elif probability >= threshold * 0.5:
            return "borderline"
        return "normal"


def get_category_for_diagnosis(diagnosis_name: str) -> str:
    """Get the category for a given diagnosis name."""
    for category, diagnoses in ECG_CATEGORIES.items():
        if diagnosis_name in diagnoses:
            return category
    return "Other diagnoses"


def parse_probabilities_csv(csv_path: str, model_type: str = "multi_label") -> Dict[str, Any]:
    """
    Parse the probabilities CSV file output from Docker analysis.

    Args:
        csv_path: Path to the probabilities CSV file
        model_type: "multi_label" for 77 classes or "binary" for LVEF/AF

    Returns:
        Dict with parsed probabilities and classifications
    """
    results = {
        "diagnoses": [],
        "by_category": {},
        "summary": {
            "total": 0,
            "abnormal": 0,
            "borderline": 0,
            "normal": 0,
        }
    }

    if not os.path.exists(csv_path):
        logger.error(f"Probabilities CSV not found: {csv_path}")
        return results

    logger.info(f"Parsing probabilities CSV: {csv_path} (type: {model_type})")

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            if not rows:
                logger.warning("Empty probabilities CSV")
                return results

            # Take the first row (single ECG analysis)
            row = rows[0]
            logger.info(f"CSV columns: {list(row.keys())}")

            # Check if this is a binary model format (has 'predictions' column)
            if model_type == "binary" or "predictions" in row:
                # Binary model format: file_name, ground_truth, predictions
                logger.info("Parsing as binary model format")
                try:
                    probability = float(row.get("predictions", 0))
                except (ValueError, TypeError):
                    logger.error(f"Could not parse predictions value: {row.get('predictions')}")
                    return results

                # Determine diagnosis name from the CSV filename
                csv_basename = os.path.basename(csv_path)
                if "lvef_equal_under_40" in csv_basename or "lvef_40" in csv_basename:
                    diagnosis_name = "LVEF ≤40%"
                    threshold = 0.5  # 50% threshold for LVEF
                    category = "Fonction Ventriculaire"
                elif "lvef_under_50" in csv_basename or "lvef_50" in csv_basename:
                    diagnosis_name = "LVEF <50%"
                    threshold = 0.5
                    category = "Fonction Ventriculaire"
                elif "afib" in csv_basename:
                    diagnosis_name = "Risque FA 5 ans"
                    threshold = 0.5
                    category = "Arythmies"
                else:
                    diagnosis_name = "Prédiction"
                    threshold = 0.5
                    category = "Autre"

                # Classify
                if probability >= threshold:
                    status = "abnormal"
                elif probability >= threshold * 0.6:  # 60% of threshold = borderline
                    status = "borderline"
                else:
                    status = "normal"

                diagnosis_result = {
                    "name": diagnosis_name,
                    "probability": round(probability * 100, 2),  # Convert to percentage
                    "threshold": round(threshold * 100, 2),
                    "status": status,
                    "category": category,
                }

                results["diagnoses"].append(diagnosis_result)
                results["summary"]["total"] = 1
                results["summary"][status] = 1

                # Group by category
                results["by_category"][category] = [diagnosis_result]

                logger.info(f"Binary result: {diagnosis_name} = {probability*100:.1f}% ({status})")

            else:
                # Multi-label format: columns with _sig_model suffix
                logger.info("Parsing as multi-label model format")
                processed_diagnoses = set()

                for col_name, value in row.items():
                    if col_name == "file_name":
                        continue

                    # Only process _sig_model columns (the actual ECG signal predictions)
                    if not col_name.endswith("_sig_model"):
                        continue

                    try:
                        probability = float(value)
                    except (ValueError, TypeError):
                        continue

                    # Extract diagnosis name (remove _sig_model suffix)
                    diagnosis_name = col_name[:-len("_sig_model")]

                    if diagnosis_name in processed_diagnoses:
                        continue
                    processed_diagnoses.add(diagnosis_name)

                    # Get threshold
                    threshold = DIAGNOSIS_THRESHOLDS.get(diagnosis_name, 0.5)

                    # Check if this is a "normal" baseline finding
                    is_normal_finding = diagnosis_name in NORMAL_BASELINE_FINDINGS

                    # Classify
                    status = classify_risk(probability, threshold, is_normal_finding)

                    # Get category
                    category = get_category_for_diagnosis(diagnosis_name)

                    diagnosis_result = {
                        "name": diagnosis_name,
                        "probability": round(probability * 100, 2),
                        "threshold": round(threshold * 100, 2),
                        "status": status,
                        "category": category,
                        "raw_column": col_name,
                    }

                    results["diagnoses"].append(diagnosis_result)
                    results["summary"]["total"] += 1
                    results["summary"][status] += 1

                    # Group by category
                    if category not in results["by_category"]:
                        results["by_category"][category] = []
                    results["by_category"][category].append(diagnosis_result)

        # Sort diagnoses by probability (highest first)
        results["diagnoses"].sort(key=lambda x: x["probability"], reverse=True)

        # Sort each category by probability
        for category in results["by_category"]:
            results["by_category"][category].sort(
                key=lambda x: x["probability"], reverse=True
            )

        logger.info(f"Parsed {results['summary']['total']} diagnoses: "
                   f"{results['summary']['abnormal']} abnormal, "
                   f"{results['summary']['borderline']} borderline, "
                   f"{results['summary']['normal']} normal")

    except Exception as e:
        logger.error(f"Error parsing probabilities CSV: {e}", exc_info=True)

    return results


async def run_single_model(
    model_id: str,
    csv_path: str,
    workspace_path: str,
    use_gpu: bool = True,
    timeout: int = 600
) -> Dict[str, Any]:
    """
    Run a single ECG analysis model via Docker.

    Args:
        model_id: ID of the model to run (e.g., "efficientnet_77")
        csv_path: Path to the input CSV file (inside Docker /data)
        workspace_path: Local workspace path
        use_gpu: Whether to use GPU
        timeout: Command timeout in seconds

    Returns:
        Dict with success status and results
    """
    if model_id not in AVAILABLE_MODELS:
        return {"success": False, "error": f"Unknown model: {model_id}"}

    model_config = AVAILABLE_MODELS[model_id]
    device = "cuda" if use_gpu else "cpu"

    # Check container status
    status = await get_docker_status()
    if not status.container_running:
        return {"success": False, "error": "AI Engine container is not running"}

    # Build docker exec command
    # Note: HeartWise determines which diagnosis to run based on CSV columns
    # No --diagnosis argument exists - it's automatic based on column names
    cmd = [
        "docker", "exec", CONTAINER_NAME,
        "python", "/app/main.py",
        "--mode", "analysis",
        "--data_path", csv_path,
        "--output_folder", "/data/outputs",
        "--ecg_signals_path", "/data/preprocessing",
        "--preprocessing_folder", "/data/preprocessing",
        "--batch_size", "1",
        "--diagnosis_classifier_device", device,
        "--signal_processing_device", device,
        "--hugging_face_api_key_path", "/data/api_key.json",
        "--use_wcr", str(model_config["use_wcr"]),
        "--use_efficientnet", str(model_config["use_efficientnet"])
    ]

    logger.info(f"Running model {model_id}: {' '.join(cmd)}")

    code, stdout, stderr = _run_command(cmd, timeout=timeout)

    if code != 0:
        error_msg = stderr or stdout or "Unknown error"
        logger.error(f"Model {model_id} failed: {error_msg}")
        return {
            "success": False,
            "error": error_msg,
            "model_id": model_id,
            "model_name": model_config["name"]
        }

    # Find and parse output file
    output_dir = os.path.join(workspace_path, "outputs")
    logger.info(f"Looking for output files in: {output_dir}")

    if not os.path.exists(output_dir):
        logger.error(f"Output directory does not exist: {output_dir}")
        return {
            "success": True,
            "warning": f"Output directory not found: {output_dir}",
            "model_id": model_id,
            "model_name": model_config["name"],
            "output": stdout
        }

    try:
        all_files = os.listdir(output_dir)
        logger.info(f"Files in output directory: {all_files}")
        prob_files = [f for f in all_files if "probabilities" in f and f.endswith(".csv")]
        logger.info(f"Probability files found: {prob_files}")
    except Exception as e:
        logger.error(f"Error listing output directory: {e}")
        return {
            "success": True,
            "warning": f"Error listing output directory: {e}",
            "model_id": model_id,
            "model_name": model_config["name"],
            "output": stdout
        }

    if not prob_files:
        return {
            "success": True,
            "warning": "No probabilities file found",
            "model_id": model_id,
            "model_name": model_config["name"],
            "output": stdout
        }

    # Get the most recent probabilities file
    prob_files.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)
    prob_path = os.path.join(output_dir, prob_files[0])
    logger.info(f"Using probabilities file: {prob_path}")

    # Parse results
    parsed_results = parse_probabilities_csv(prob_path, model_config["type"])
    logger.info(f"Parsed results summary: {parsed_results.get('summary', {})}")

    return {
        "success": True,
        "model_id": model_id,
        "model_name": model_config["name"],
        "model_type": model_config["type"],
        "architecture": model_config["architecture"],
        "results": parsed_results,
        "output_file": prob_files[0]
    }


async def run_preprocessing(
    csv_filename: str,
    workspace_path: str,
    use_gpu: bool = False,
    timeout: int = 300
) -> Dict[str, Any]:
    """
    Run ECG preprocessing to convert XML to base64.

    Args:
        csv_filename: Name of the CSV file in inputs folder (e.g., "preprocess_PAT001.csv")
        workspace_path: Local workspace path
        use_gpu: Whether to use GPU
        timeout: Command timeout in seconds

    Returns:
        Dict with success status
    """
    device = "cuda" if use_gpu else "cpu"

    # Check container status
    status = await get_docker_status()
    if not status.container_running:
        return {"success": False, "error": "AI Engine container is not running"}

    # The CSV file should already exist in inputs folder
    csv_path = f"/data/inputs/{csv_filename}"

    cmd = [
        "docker", "exec", CONTAINER_NAME,
        "python", "/app/main.py",
        "--mode", "preprocessing",
        "--data_path", csv_path,
        "--output_folder", "/data/outputs",
        "--ecg_signals_path", "/data/ecg_signals",
        "--preprocessing_folder", "/data/preprocessing",
        "--batch_size", "1",
        "--preprocessing_n_workers", "1",
        "--diagnosis_classifier_device", device,
        "--signal_processing_device", device,
        "--hugging_face_api_key_path", "/data/api_key.json",
        "--use_wcr", "False",
        "--use_efficientnet", "False"
    ]

    logger.info(f"Running preprocessing: {' '.join(cmd)}")

    code, stdout, stderr = _run_command(cmd, timeout=timeout)

    if code != 0:
        error_msg = stderr or stdout or "Preprocessing failed"
        logger.error(f"Preprocessing failed: {error_msg}")
        return {"success": False, "error": error_msg}

    return {"success": True, "output": stdout}


async def run_full_ecg_analysis(
    ecg_file_path: str,
    patient_id: str,
    workspace_path: str,
    selected_models: List[str],
    use_gpu: bool = True
) -> Dict[str, Any]:
    """
    Run complete ECG analysis pipeline with selected models.

    Args:
        ecg_file_path: Path to the ECG file (XML)
        patient_id: Patient identifier
        workspace_path: Local workspace path
        selected_models: List of model IDs to run, or ["all"] for all models
        use_gpu: Whether to use GPU

    Returns:
        Comprehensive analysis results
    """
    import time
    start_time = time.time()

    results = {
        "success": False,
        "patient_id": patient_id,
        "ecg_filename": os.path.basename(ecg_file_path),
        "models_executed": [],
        "results": {},
        "summary": {
            "overall_status": "normal",
            "total_abnormal": 0,
            "total_borderline": 0,
            "critical_findings": [],
        },
        "warnings": [],
        "processing_time_ms": 0
    }

    # Clean up old files from previous analyses to avoid cache issues
    # Note: Don't clean ecg_signals here - it contains the file just uploaded by api.py
    for dir_name in ["preprocessing", "outputs", "inputs"]:
        dir_path = os.path.join(workspace_path, dir_name)
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                logger.info(f"Cleaned up old {dir_name} directory")
            except Exception as e:
                logger.warning(f"Failed to clean {dir_name}: {e}")
        os.makedirs(dir_path, exist_ok=True)

    # Create batch_1 subdirectory in outputs (required by Docker HeartWise)
    batch_dir = os.path.join(workspace_path, "outputs", "batch_1")
    os.makedirs(batch_dir, exist_ok=True)

    # NOTE: 8-to-12 lead expansion is NOT needed - Docker HeartWise handles this internally
    # The Docker can process 8-lead files (I, II, V1-V6) and calculates derived leads itself
    # Keeping expand_8_to_12_leads function for potential future use but not calling it here

    # Determine which models to run
    if "all" in selected_models:
        models_to_run = list(AVAILABLE_MODELS.keys())
    else:
        models_to_run = [m for m in selected_models if m in AVAILABLE_MODELS]

    if not models_to_run:
        results["success"] = False
        results["error"] = "No valid models selected"
        return results

    # Step 1: Prepare input CSV
    ecg_basename = os.path.splitext(os.path.basename(ecg_file_path))[0]

    # Create input CSV for preprocessing
    inputs_dir = os.path.join(workspace_path, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)

    # Prepare base64 filename for analysis
    base64_filename = f"{ecg_basename}.base64"

    # Create a simple CSV for preprocessing (only needs patient_id and file columns)
    preprocess_csv_filename = f"preprocess_{patient_id}.csv"
    preprocess_csv_path = os.path.join(inputs_dir, preprocess_csv_filename)

    with open(preprocess_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "ecg_machine_diagnosis", "77_classes_ecg_file_name"])
        writer.writerow([patient_id, "unknown", os.path.basename(ecg_file_path)])

    # Step 2: Check if preprocessing is needed
    preprocessing_dir = os.path.join(workspace_path, "preprocessing")
    base64_path = os.path.join(preprocessing_dir, f"{ecg_basename}.base64")

    if not os.path.exists(base64_path):
        # Copy ECG file to ecg_signals folder
        ecg_signals_dir = os.path.join(workspace_path, "ecg_signals")
        os.makedirs(ecg_signals_dir, exist_ok=True)

        dest_ecg_path = os.path.join(ecg_signals_dir, os.path.basename(ecg_file_path))
        if not os.path.exists(dest_ecg_path):
            shutil.copy2(ecg_file_path, dest_ecg_path)

        # Run preprocessing
        preprocess_result = await run_preprocessing(
            preprocess_csv_filename, workspace_path, use_gpu=False
        )
        if not preprocess_result["success"]:
            results["error"] = f"Preprocessing failed: {preprocess_result.get('error')}"
            return results

    # Step 3: Run each selected model with its OWN CSV file
    # Docker HeartWise uses the first diagnosis column it finds, so we create
    # a separate CSV for each diagnosis type with ONLY that column

    for model_id in models_to_run:
        logger.info(f"Running model: {model_id}")

        model_config = AVAILABLE_MODELS[model_id]
        diag_type = model_config["docker_diagnosis"]

        # Create model-specific CSV with ONLY the required diagnosis column
        model_csv_filename = f"analysis_{patient_id}_{model_id}.csv"
        model_csv_path = os.path.join(inputs_dir, model_csv_filename)

        with open(model_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if diag_type == "ecg_machine_diagnosis":
                # 77-class model uses string diagnosis
                writer.writerow(["patient_id", "ecg_machine_diagnosis", "77_classes_ecg_file_name"])
                writer.writerow([patient_id, "unknown", base64_filename])
            elif diag_type == "lvef_40":
                # Binary LVEF model uses numeric value (0 = unknown/to predict)
                writer.writerow(["patient_id", "lvef_40", "lvef_40_ecg_file_name"])
                writer.writerow([patient_id, 0, base64_filename])
            elif diag_type == "lvef_50":
                writer.writerow(["patient_id", "lvef_50", "lvef_50_ecg_file_name"])
                writer.writerow([patient_id, 0, base64_filename])
            elif diag_type == "afib_5y":
                writer.writerow(["patient_id", "afib_5y", "afib_ecg_file_name"])
                writer.writerow([patient_id, 0, base64_filename])
            else:
                # Default case
                writer.writerow(["patient_id", diag_type, f"{diag_type}_ecg_file_name"])
                writer.writerow([patient_id, "unknown", base64_filename])

        docker_csv_path = f"/data/inputs/{model_csv_filename}"

        model_result = await run_single_model(
            model_id=model_id,
            csv_path=docker_csv_path,
            workspace_path=workspace_path,
            use_gpu=use_gpu
        )

        results["models_executed"].append(model_id)
        results["results"][model_id] = model_result

        # Update summary
        if model_result.get("success") and "results" in model_result:
            model_summary = model_result["results"].get("summary", {})
            results["summary"]["total_abnormal"] += model_summary.get("abnormal", 0)
            results["summary"]["total_borderline"] += model_summary.get("borderline", 0)

            # Collect critical findings (abnormal results)
            for diag in model_result["results"].get("diagnoses", []):
                if diag["status"] == "abnormal":
                    results["summary"]["critical_findings"].append({
                        "diagnosis": diag["name"],
                        "probability": diag["probability"],
                        "model": model_id
                    })

    # Determine overall status
    if results["summary"]["total_abnormal"] > 0:
        results["summary"]["overall_status"] = "abnormal"
    elif results["summary"]["total_borderline"] > 0:
        results["summary"]["overall_status"] = "borderline"
    else:
        results["summary"]["overall_status"] = "normal"

    results["success"] = True
    results["processing_time_ms"] = int((time.time() - start_time) * 1000)

    return results
