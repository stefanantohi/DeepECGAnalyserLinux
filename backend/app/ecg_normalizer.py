"""
ECG XML Format Normalizer.

Converts various ECG XML formats to GE MUSE RestingECG format
that HeartWise Docker can process.

HeartWise Docker expects GE MUSE XML with TWO Waveform sections:
- Index 0: Median (representative beat, ~600 samples)
- Index 1: Rhythm (10-second strip, ~5000 samples at 500Hz)

================================================================================
SUPPORTED ECG FORMATS
================================================================================

Format               | Detection               | Conversion    | Viewer  | Notes
---------------------|--------------------------|---------------|---------|------
GE MUSE (2 Waveform) | <RestingECG> + 2x<Waveform> | Natif (aucune) | Natif   | Format cible HeartWise
GE MUSE (1 Waveform) | <RestingECG> + 1x<Waveform> | Auto: +Median  | Natif   | Section Median generee par duplication
Philips PageWriter   | ns philips.com/restingecgdata | philips_converter.py | repbeats | Repbeats base64 int16 -> MUSE
HL7 aECG (FDA)       | <AnnotatedECG> ns hl7-org:v3 | ecg_normalizer.py | digits   | digits espaces, scale uV, 500Hz
CardiologyXML        | <CardiologyXML>           | ecg_normalizer.py | base64  | <LeadData lead="X"> base64 int16

Pipeline de conversion:
  1. detect_and_convert_encoding() - UTF-16/Latin-1 -> UTF-8
  2. auto_convert_if_philips()     - Philips -> GE MUSE (philips_converter.py)
  3. normalize_to_muse()           - Autres formats -> GE MUSE (ce fichier)

Ajout d'un nouveau format:
  1. Ajouter la detection dans detect_ecg_format()
  2. Creer la fonction _convert_<format>()
  3. Ajouter le branchement dans normalize_to_muse()
  4. Ajouter le parsing viewer dans api.py:_parse_xml_waveform()
  5. Mettre a jour SUPPORTED_FORMATS ci-dessous
================================================================================
"""

import base64
import logging
import re
import struct
import xml.etree.ElementTree as ET
import zlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 8-lead order (leads stored by GE MUSE, derived leads are calculated by HeartWise)
LEAD_ORDER_8 = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Registry of supported ECG formats (used by API for user-facing info)
SUPPORTED_FORMATS = [
    {
        "id": "ge_muse",
        "name": "GE MUSE RestingECG",
        "vendor": "GE Healthcare",
        "detection": "<RestingECG> avec sections <Waveform>",
        "conversion": "Natif (format cible)",
        "data_encoding": "Base64 int16 little-endian",
        "native": True,
    },
    {
        "id": "philips",
        "name": "Philips PageWriter / TC",
        "vendor": "Philips Healthcare",
        "detection": "Namespace medical.philips.com, <restingecgdata>",
        "conversion": "Repbeats extraits, resampling 1000->500Hz, extension 10s",
        "data_encoding": "Base64 int16 (repbeats) + XLI compresse (parsedwaveforms)",
        "native": False,
    },
    {
        "id": "hl7_aecg",
        "name": "HL7 aECG (Annotated ECG)",
        "vendor": "Standard FDA / HL7",
        "detection": "<AnnotatedECG> namespace urn:hl7-org:v3",
        "conversion": "Digits entiers -> uV (scale) -> MUSE int16 (/ 4.88)",
        "data_encoding": "Entiers espaces dans <digits>, scale + origin en uV",
        "native": False,
    },
    {
        "id": "cardiology_xml",
        "name": "CardiologyXML",
        "vendor": "Format generique",
        "detection": "<CardiologyXML> avec <LeadData lead='X'>",
        "conversion": "Decodage base64 int16, construction MUSE XML",
        "data_encoding": "Base64 int16 little-endian dans <WaveformData>",
        "native": False,
    },
]


def detect_ecg_format(xml_path: str) -> str:
    """
    Detect the ECG XML format of a file.

    Returns:
        One of: "ge_muse_2waveform", "ge_muse_1waveform", "cardiology_xml",
                "philips", "unknown"
    """
    try:
        content = None
        for encoding in ['utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'latin-1']:
            try:
                with open(xml_path, 'r', encoding=encoding) as f:
                    content = f.read(8000)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        if content is None:
            return "unknown"

        content_lower = content.lower()

        # Check for HL7 aECG (Annotated ECG, FDA standard)
        if 'urn:hl7-org:v3' in content or '<annotatedecg' in content_lower:
            return "hl7_aecg"

        # Check for Philips (handled separately by philips_converter)
        if ('philips' in content_lower or 'pagewriter' in content_lower or
            'restingecgdata' in content_lower or
            ('repbeats' in content_lower and 'samplespersec' in content_lower)):
            return "philips"

        # Check for CardiologyXML format
        if '<cardiologyxml' in content_lower:
            return "cardiology_xml"

        # Check for GE MUSE RestingECG
        if '<restingecg' in content_lower:
            # Count Waveform sections
            waveform_count = content_lower.count('<waveform>')
            if waveform_count == 0:
                # Need to read more of the file
                with open(xml_path, 'r', encoding='utf-8') as f:
                    full_content = f.read()
                waveform_count = full_content.lower().count('<waveform>')

            if waveform_count >= 2:
                return "ge_muse_2waveform"
            elif waveform_count == 1:
                return "ge_muse_1waveform"
            else:
                return "ge_muse_1waveform"  # RestingECG but no Waveform found yet

        return "unknown"

    except Exception as e:
        logger.error(f"Error detecting ECG format: {e}")
        return "unknown"


def _decode_waveform_base64(b64_data: str) -> List[int]:
    """Decode base64-encoded 16-bit signed integer waveform data."""
    try:
        clean_b64 = ''.join(b64_data.split())
        raw_bytes = base64.b64decode(clean_b64)
        num_samples = len(raw_bytes) // 2
        samples = list(struct.unpack(f'<{num_samples}h', raw_bytes))
        return samples
    except Exception as e:
        logger.error(f"Error decoding waveform base64: {e}")
        return []


def _encode_waveform_base64(samples: List[int]) -> str:
    """Encode waveform samples to base64 with line breaks."""
    raw_bytes = struct.pack(f'<{len(samples)}h', *samples)
    b64_data = base64.b64encode(raw_bytes).decode('ascii')
    lines = [b64_data[i:i+68] for i in range(0, len(b64_data), 68)]
    return '\n'.join(lines)


def _build_muse_xml(
    lead_data: Dict[str, List[int]],
    sample_rate: int = 500,
    patient_id: str = "",
    acquisition_datetime: str = "",
) -> str:
    """
    Build a complete GE MUSE RestingECG XML with 2 Waveform sections.

    Args:
        lead_data: Dict of lead_name -> samples (10s rhythm data)
        sample_rate: Sampling rate in Hz
        patient_id: Patient identifier
        acquisition_datetime: Acquisition date/time string
    """
    # Calculate sample counts
    rhythm_samples = len(next(iter(lead_data.values()))) if lead_data else 5000
    median_samples = 600  # ~1.2s representative beat at 500Hz

    # Filter to 8 leads only (HeartWise calculates derived leads)
    leads_8 = {}
    for lead_name in LEAD_ORDER_8:
        if lead_name in lead_data:
            leads_8[lead_name] = lead_data[lead_name]

    if len(leads_8) < 2:
        logger.error(f"Insufficient leads for MUSE conversion: {list(leads_8.keys())}")
        return ""

    # Parse acquisition date/time
    acq_date = datetime.now().strftime('%m-%d-%Y')
    acq_time = datetime.now().strftime('%H:%M:%S')
    if acquisition_datetime:
        try:
            dt = datetime.fromisoformat(acquisition_datetime.replace('Z', '+00:00'))
            acq_date = dt.strftime('%m-%d-%Y')
            acq_time = dt.strftime('%H:%M:%S')
        except (ValueError, AttributeError):
            pass

    xml_parts = []
    xml_parts.append('<?xml version="1.0" encoding="utf-8"?>')
    xml_parts.append('<!DOCTYPE RestingECG SYSTEM "restecg.dtd">')
    xml_parts.append('<RestingECG>')

    # MuseInfo
    xml_parts.append('   <MuseInfo>')
    xml_parts.append('      <MuseVersion>10.1.0.0</MuseVersion>')
    xml_parts.append('   </MuseInfo>')

    # PatientDemographics
    xml_parts.append('   <PatientDemographics>')
    xml_parts.append(f'      <PatientID>{patient_id}</PatientID>')
    xml_parts.append('   </PatientDemographics>')

    # TestDemographics
    xml_parts.append('   <TestDemographics>')
    xml_parts.append(f'      <AcquisitionDate>{acq_date}</AcquisitionDate>')
    xml_parts.append(f'      <AcquisitionTime>{acq_time}</AcquisitionTime>')
    xml_parts.append('   </TestDemographics>')

    # RestingECGMeasurements
    xml_parts.append('   <RestingECGMeasurements>')
    xml_parts.append('      <VentricularRate>0</VentricularRate>')
    xml_parts.append('      <QRSDuration>0</QRSDuration>')
    xml_parts.append('      <QTInterval>0</QTInterval>')
    xml_parts.append(f'      <ECGSampleBase>{sample_rate}</ECGSampleBase>')
    xml_parts.append('      <ECGSampleExponent>0</ECGSampleExponent>')
    xml_parts.append('   </RestingECGMeasurements>')

    # Helper to build a Waveform section
    def add_waveform_section(waveform_type, sample_type, target_samples):
        byte_count = target_samples * 2
        xml_parts.append('   <Waveform>')
        xml_parts.append(f'      <WaveformType>{waveform_type}</WaveformType>')
        xml_parts.append('      <WaveformStartTime>0</WaveformStartTime>')
        xml_parts.append(f'      <NumberofLeads>{len(leads_8)}</NumberofLeads>')
        xml_parts.append(f'      <SampleType>{sample_type}</SampleType>')
        xml_parts.append(f'      <SampleBase>{sample_rate}</SampleBase>')
        xml_parts.append('      <SampleExponent>0</SampleExponent>')
        xml_parts.append('      <HighPassFilter>5</HighPassFilter>')
        xml_parts.append('      <LowPassFilter>150</LowPassFilter>')
        xml_parts.append('      <ACFilter>60</ACFilter>')

        for lead_name in LEAD_ORDER_8:
            if lead_name not in leads_8:
                continue
            samples = leads_8[lead_name][:target_samples]
            # Pad if shorter than target
            if len(samples) < target_samples:
                samples = samples + samples * ((target_samples // len(samples)) + 1)
                samples = samples[:target_samples]

            b64_data = _encode_waveform_base64(samples)
            crc = zlib.crc32(struct.pack(f'<{len(samples)}h', *samples)) & 0xffffffff

            xml_parts.append('      <LeadData>')
            xml_parts.append(f'         <LeadByteCountTotal>{byte_count}</LeadByteCountTotal>')
            xml_parts.append('         <LeadTimeOffset>0</LeadTimeOffset>')
            xml_parts.append(f'         <LeadSampleCountTotal>{target_samples}</LeadSampleCountTotal>')
            xml_parts.append('         <LeadAmplitudeUnitsPerBit>4.88</LeadAmplitudeUnitsPerBit>')
            xml_parts.append('         <LeadAmplitudeUnits>MICROVOLTS</LeadAmplitudeUnits>')
            xml_parts.append('         <LeadHighLimit>32767</LeadHighLimit>')
            xml_parts.append('         <LeadLowLimit>-32768</LeadLowLimit>')
            xml_parts.append(f'         <LeadID>{lead_name}</LeadID>')
            xml_parts.append('         <LeadOffsetFirstSample>0</LeadOffsetFirstSample>')
            xml_parts.append('         <FirstSampleBaseline>0</FirstSampleBaseline>')
            xml_parts.append('         <LeadSampleSize>2</LeadSampleSize>')
            xml_parts.append('         <LeadOff>FALSE</LeadOff>')
            xml_parts.append('         <BaselineSway>FALSE</BaselineSway>')
            xml_parts.append(f'         <LeadDataCRC32>{crc}</LeadDataCRC32>')
            xml_parts.append('         <WaveFormData>')
            xml_parts.append(b64_data)
            xml_parts.append('         </WaveFormData>')
            xml_parts.append('      </LeadData>')

        xml_parts.append('   </Waveform>')

    # Waveform 0: Median (representative beat)
    add_waveform_section('Median', 'REPRESENTATIVE_BEAT', median_samples)

    # Waveform 1: Rhythm (10-second strip)
    add_waveform_section('Rhythm', 'CONTINUOUS_SAMPLES', rhythm_samples)

    xml_parts.append('</RestingECG>')

    return '\n'.join(xml_parts)


def _convert_cardiology_xml(xml_path: str) -> Tuple[bool, str]:
    """
    Convert CardiologyXML format to GE MUSE RestingECG.

    CardiologyXML structure:
    <CardiologyXML>
      <PatientDemographics><PatientID>...</PatientID></PatientDemographics>
      <TestInfo><AcquisitionDateTime>...</AcquisitionDateTime><SampleRate>500</SampleRate></TestInfo>
      <Waveforms>
        <SamplingRate>500</SamplingRate>
        <NumberOfLeads>12</NumberOfLeads>
        <LeadData lead="I"><WaveformData>base64...</WaveformData></LeadData>
        ...
      </Waveforms>
    </CardiologyXML>
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Extract patient info
        patient_id = ""
        pid_elem = root.find('.//PatientID')
        if pid_elem is not None and pid_elem.text:
            patient_id = pid_elem.text.strip()

        # Extract acquisition datetime
        acq_datetime = ""
        acq_elem = root.find('.//AcquisitionDateTime')
        if acq_elem is not None and acq_elem.text:
            acq_datetime = acq_elem.text.strip()

        # Extract sample rate
        sample_rate = 500
        for tag_name in ['SamplingRate', 'SampleRate']:
            sr_elem = root.find(f'.//{tag_name}')
            if sr_elem is not None and sr_elem.text:
                try:
                    sample_rate = int(sr_elem.text)
                except ValueError:
                    pass
                break

        # Extract lead data
        lead_data = {}
        for lead_elem in root.iter('LeadData'):
            lead_name = lead_elem.get('lead', '')
            if not lead_name:
                # Try child element <LeadID>
                lid = lead_elem.find('LeadID')
                if lid is not None and lid.text:
                    lead_name = lid.text.strip()

            if not lead_name:
                continue

            waveform_elem = lead_elem.find('WaveformData')
            if waveform_elem is None or not waveform_elem.text:
                continue

            samples = _decode_waveform_base64(waveform_elem.text)
            if samples:
                lead_data[lead_name] = samples
                logger.debug(f"CardiologyXML: extracted {len(samples)} samples for lead {lead_name}")

        if not lead_data:
            return False, "No waveform data found in CardiologyXML"

        logger.info(f"CardiologyXML: extracted {len(lead_data)} leads "
                    f"({list(lead_data.keys())}), {sample_rate}Hz")

        # Ensure we have 10s of data at target rate (5000 samples at 500Hz)
        target_samples = sample_rate * 10  # 10 seconds

        # Pad/truncate leads to target length
        for lead_name in lead_data:
            samples = lead_data[lead_name]
            if len(samples) < target_samples:
                # Repeat the signal to reach target duration
                extended = []
                while len(extended) < target_samples:
                    remaining = target_samples - len(extended)
                    extended.extend(samples[:remaining])
                lead_data[lead_name] = extended[:target_samples]
            elif len(samples) > target_samples:
                lead_data[lead_name] = samples[:target_samples]

        # Build GE MUSE XML
        muse_xml = _build_muse_xml(
            lead_data=lead_data,
            sample_rate=sample_rate,
            patient_id=patient_id,
            acquisition_datetime=acq_datetime,
        )

        if not muse_xml:
            return False, "Failed to build MUSE XML"

        # Overwrite the file with the converted version
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(muse_xml)

        logger.info(f"Successfully converted CardiologyXML to GE MUSE: {xml_path}")
        return True, "Converted CardiologyXML to GE MUSE format"

    except ET.ParseError as e:
        logger.error(f"XML parse error in CardiologyXML: {e}")
        return False, f"XML parse error: {e}"
    except Exception as e:
        logger.error(f"Error converting CardiologyXML: {e}", exc_info=True)
        return False, str(e)


def _fix_single_waveform_muse(xml_path: str) -> Tuple[bool, str]:
    """
    Fix GE MUSE files that have only 1 Waveform section.

    HeartWise expects Waveform at index 1. If there's only 1 section,
    we duplicate it: index 0 = Median (truncated), index 1 = original Rhythm.
    """
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find the single Waveform section
        waveform_pattern = re.compile(
            r'(<Waveform[^>]*>)(.*?)(</Waveform>)',
            re.DOTALL | re.IGNORECASE
        )
        matches = list(waveform_pattern.finditer(content))

        if len(matches) != 1:
            return False, f"Expected 1 Waveform section, found {len(matches)}"

        match = matches[0]
        original_section = match.group(0)

        # Create a Median section by modifying the WaveformType and truncating data
        median_section = original_section

        # Change WaveformType to Median
        median_section = re.sub(
            r'<WaveformType>[^<]*</WaveformType>',
            '<WaveformType>Median</WaveformType>',
            median_section,
            flags=re.IGNORECASE
        )

        # Change SampleType
        median_section = re.sub(
            r'<SampleType>[^<]*</SampleType>',
            '<SampleType>REPRESENTATIVE_BEAT</SampleType>',
            median_section,
            flags=re.IGNORECASE
        )

        # Truncate each lead's waveform data to ~600 samples
        def truncate_lead(lead_match):
            lead_content = lead_match.group(0)

            wf_match = re.search(
                r'<WaveFormData>([^<]+)</WaveFormData>',
                lead_content,
                re.IGNORECASE
            )
            if not wf_match:
                return lead_content

            b64_data = wf_match.group(1)
            samples = _decode_waveform_base64(b64_data)
            if not samples:
                return lead_content

            # Truncate to 600 samples
            truncated = samples[:600]
            new_b64 = _encode_waveform_base64(truncated)

            # Update byte count and sample count
            new_byte_count = len(truncated) * 2
            lead_content = re.sub(
                r'<LeadByteCountTotal>\d+</LeadByteCountTotal>',
                f'<LeadByteCountTotal>{new_byte_count}</LeadByteCountTotal>',
                lead_content, flags=re.IGNORECASE
            )
            lead_content = re.sub(
                r'<LeadSampleCountTotal>\d+</LeadSampleCountTotal>',
                f'<LeadSampleCountTotal>{len(truncated)}</LeadSampleCountTotal>',
                lead_content, flags=re.IGNORECASE
            )
            # Update CRC
            crc = zlib.crc32(struct.pack(f'<{len(truncated)}h', *truncated)) & 0xffffffff
            lead_content = re.sub(
                r'<LeadDataCRC32>\d+</LeadDataCRC32>',
                f'<LeadDataCRC32>{crc}</LeadDataCRC32>',
                lead_content, flags=re.IGNORECASE
            )
            # Replace waveform data
            lead_content = re.sub(
                r'<WaveFormData>[^<]+</WaveFormData>',
                f'<WaveFormData>{new_b64}</WaveFormData>',
                lead_content, flags=re.IGNORECASE
            )
            return lead_content

        lead_pattern = re.compile(r'<LeadData>.*?</LeadData>', re.DOTALL | re.IGNORECASE)
        median_section = lead_pattern.sub(truncate_lead, median_section)

        # Insert Median BEFORE the original Rhythm section
        new_content = content[:match.start()] + median_section + '\n' + original_section + content[match.end():]

        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        logger.info(f"Added Median Waveform section to single-waveform MUSE: {xml_path}")
        return True, "Added Median Waveform section (was single-waveform)"

    except Exception as e:
        logger.error(f"Error fixing single-waveform MUSE: {e}", exc_info=True)
        return False, str(e)


def _convert_hl7_aecg(xml_path: str) -> Tuple[bool, str]:
    """
    Convert HL7 aECG (Annotated ECG / FDA standard) format to GE MUSE RestingECG.

    HL7 aECG structure:
    <AnnotatedECG xmlns="urn:hl7-org:v3">
      ...
      <component><series><component><sequenceSet>
        <component><sequence>  (TIME_ABSOLUTE with <increment>)
        <component><sequence>  (MDC_ECG_LEAD_X with <digits>)
        ...
      </sequenceSet></component></series></component>
    </AnnotatedECG>

    Lead data in <digits> as space-separated integers.
    Scale in <scale unit="uV" value="1.250000"/>.
    Sample rate from <increment unit="s" value="0.002000"/> = 500Hz.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Handle HL7 namespace
        ns = ''
        root_tag = root.tag
        if '}' in root_tag:
            ns = root_tag.split('}')[0] + '}'

        def ns_tag(tag):
            return f'{ns}{tag}'

        # Extract patient ID
        patient_id = ""
        for id_elem in root.iter(ns_tag('id')):
            ext = id_elem.get('extension', '')
            if ext and not ext.startswith('-'):
                patient_id = ext
                break

        # Extract acquisition datetime from effectiveTime
        acq_datetime = ""
        for et_elem in root.iter(ns_tag('effectiveTime')):
            low = et_elem.find(ns_tag('low'))
            if low is not None:
                val = low.get('value', '')
                if len(val) >= 14:
                    try:
                        dt = datetime.strptime(val[:14], '%Y%m%d%H%M%S')
                        acq_datetime = dt.isoformat()
                    except ValueError:
                        pass
            if acq_datetime:
                break

        # Parse lead data from sequenceSet components
        lead_data = {}
        sample_rate = 500
        muse_units_per_bit = 4.88

        for sequence in root.iter(ns_tag('sequence')):
            code_elem = sequence.find(ns_tag('code'))
            if code_elem is None:
                continue

            code_val = code_elem.get('code', '')

            # Extract sample rate from TIME_ABSOLUTE sequence
            if code_val == 'TIME_ABSOLUTE':
                value_elem = sequence.find(ns_tag('value'))
                if value_elem is not None:
                    inc_elem = value_elem.find(ns_tag('increment'))
                    if inc_elem is not None:
                        try:
                            inc_val = float(inc_elem.get('value', '0.002'))
                            if inc_val > 0:
                                sample_rate = int(round(1.0 / inc_val))
                        except (ValueError, ZeroDivisionError):
                            pass
                continue

            # Extract lead data from MDC_ECG_LEAD_* sequences
            if not code_val.startswith('MDC_ECG_LEAD_'):
                continue

            # Map MDC lead code to standard name
            lead_name = code_val.replace('MDC_ECG_LEAD_', '')

            value_elem = sequence.find(ns_tag('value'))
            if value_elem is None:
                continue

            # Get scale factor
            scale_val = 1.0
            scale_elem = value_elem.find(ns_tag('scale'))
            if scale_elem is not None:
                try:
                    scale_val = float(scale_elem.get('value', '1.0'))
                except ValueError:
                    pass

            # Get origin offset
            origin_val = 0.0
            origin_elem = value_elem.find(ns_tag('origin'))
            if origin_elem is not None:
                try:
                    origin_val = float(origin_elem.get('value', '0'))
                except ValueError:
                    pass

            # Extract digits (space-separated integers)
            digits_elem = value_elem.find(ns_tag('digits'))
            if digits_elem is None or not digits_elem.text:
                continue

            try:
                raw_digits = [int(d) for d in digits_elem.text.strip().split()]
            except ValueError:
                logger.warning(f"HL7 aECG: could not parse digits for lead {lead_name}")
                continue

            # Convert to MUSE int16 units:
            # actual_uV = raw_digit * scale + origin
            # muse_digit = actual_uV / muse_units_per_bit
            samples = [int(round((d * scale_val + origin_val) / muse_units_per_bit))
                       for d in raw_digits]

            # Clamp to int16 range
            samples = [max(-32768, min(32767, s)) for s in samples]

            lead_data[lead_name] = samples
            logger.debug(f"HL7 aECG: extracted {len(samples)} samples for lead {lead_name}")

        if not lead_data:
            return False, "No waveform data found in HL7 aECG"

        logger.info(f"HL7 aECG: extracted {len(lead_data)} leads "
                    f"({list(lead_data.keys())}), {sample_rate}Hz")

        # Ensure 10s of data
        target_samples = sample_rate * 10
        for lead_name in lead_data:
            samples = lead_data[lead_name]
            if len(samples) < target_samples:
                extended = []
                while len(extended) < target_samples:
                    remaining = target_samples - len(extended)
                    extended.extend(samples[:remaining])
                lead_data[lead_name] = extended[:target_samples]
            elif len(samples) > target_samples:
                lead_data[lead_name] = samples[:target_samples]

        # Build GE MUSE XML
        muse_xml = _build_muse_xml(
            lead_data=lead_data,
            sample_rate=sample_rate,
            patient_id=patient_id,
            acquisition_datetime=acq_datetime,
        )

        if not muse_xml:
            return False, "Failed to build MUSE XML from HL7 aECG"

        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(muse_xml)

        logger.info(f"Successfully converted HL7 aECG to GE MUSE: {xml_path}")
        return True, "Converted HL7 aECG to GE MUSE format"

    except ET.ParseError as e:
        logger.error(f"XML parse error in HL7 aECG: {e}")
        return False, f"XML parse error: {e}"
    except Exception as e:
        logger.error(f"Error converting HL7 aECG: {e}", exc_info=True)
        return False, str(e)


def normalize_to_muse(xml_path: str) -> Tuple[bool, str, str]:
    """
    Detect ECG format and normalize to GE MUSE with 2 Waveform sections.

    This function should be called AFTER the Philips converter has already run.
    It handles non-Philips formats that are not already valid GE MUSE.

    Args:
        xml_path: Path to the ECG XML file

    Returns:
        Tuple of (was_converted, format_detected, message)
    """
    try:
        fmt = detect_ecg_format(xml_path)
        logger.info(f"Detected ECG format: {fmt} for {xml_path}")

        if fmt == "ge_muse_2waveform":
            # Already valid, no conversion needed
            return False, "ge_muse", "Already valid GE MUSE format with 2 Waveform sections"

        if fmt == "hl7_aecg":
            success, msg = _convert_hl7_aecg(xml_path)
            return success, "hl7_aecg", msg

        if fmt == "philips":
            # Should have been handled by philips_converter already
            return False, "philips", "Philips format (should be handled by philips_converter)"

        if fmt == "cardiology_xml":
            success, msg = _convert_cardiology_xml(xml_path)
            return success, "cardiology_xml", msg

        if fmt == "ge_muse_1waveform":
            success, msg = _fix_single_waveform_muse(xml_path)
            return success, "ge_muse", msg

        # Unknown format - try to parse as generic XML with lead data
        logger.warning(f"Unknown ECG format for {xml_path}, attempting generic conversion")
        return False, "unknown", "Unknown ECG format, no conversion applied"

    except Exception as e:
        logger.error(f"Error normalizing ECG XML: {e}", exc_info=True)
        return False, "unknown", f"Error: {str(e)}"
