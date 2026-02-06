"""
Philips PageWriter TC XML to GE MUSE RestingECG XML Converter.

This module converts Philips ECG XML files (PageWriter TC format) to GE MUSE
RestingECG format that HeartWise can process.

Philips PageWriter TC uses:
- XLI compression for main waveforms (proprietary)
- Uncompressed Base64 for representative beats (<repbeats>)

This converter extracts the representative beats and creates a GE MUSE-compatible
XML file that can be processed by HeartWise.
"""

import base64
import logging
import struct
import xml.etree.ElementTree as ET
import zlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 8-lead order (leads that GE MUSE stores, others are derived)
LEAD_ORDER_8 = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def detect_philips_pagewriter(xml_path: str) -> bool:
    """
    Detect if an XML file is in Philips PageWriter TC format.

    Args:
        xml_path: Path to the XML file

    Returns:
        True if the file is Philips PageWriter format
    """
    try:
        content = None
        # Try multiple encodings
        for encoding in ['utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'latin-1']:
            try:
                with open(xml_path, 'r', encoding=encoding) as f:
                    content = f.read(8000)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        if content is None:
            logger.warning(f"Could not read file with any encoding: {xml_path}")
            return False

        content_lower = content.lower()

        # Check for Philips-specific markers
        is_philips = (
            'www3.medical.philips.com' in content_lower or
            'philipsecg' in content_lower or
            'philips.com' in content_lower or
            '<restingecgdata' in content_lower or
            'pagewriter' in content_lower or
            ('<repbeats' in content_lower and 'samplespersec' in content_lower)
        )

        if is_philips:
            logger.info(f"Detected Philips PageWriter format in {xml_path}")

        return is_philips
    except Exception as e:
        logger.error(f"Error detecting Philips format: {e}")
        return False


def decode_philips_waveform(base64_data: str) -> List[int]:
    """
    Decode Philips Base64 encoded waveform data.

    The Philips format stores 16-bit signed integers in Base64.

    Args:
        base64_data: Base64 encoded waveform string

    Returns:
        List of integer sample values
    """
    try:
        clean_b64 = ''.join(base64_data.split())
        raw_bytes = base64.b64decode(clean_b64)
        num_samples = len(raw_bytes) // 2
        samples = list(struct.unpack(f'<{num_samples}h', raw_bytes))
        return samples
    except Exception as e:
        logger.error(f"Error decoding Philips waveform: {e}")
        return []


def resample_waveform(samples: List[int], from_rate: int, to_rate: int) -> List[int]:
    """Resample waveform using linear interpolation."""
    if from_rate == to_rate:
        return samples

    if len(samples) < 2:
        return samples

    ratio = to_rate / from_rate
    new_length = int(len(samples) * ratio)
    resampled = []

    for i in range(new_length):
        orig_pos = i / ratio
        idx = int(orig_pos)
        frac = orig_pos - idx

        if idx + 1 < len(samples):
            value = samples[idx] * (1 - frac) + samples[idx + 1] * frac
        else:
            value = samples[-1]

        resampled.append(int(round(value)))

    return resampled


def extend_to_duration(samples: List[int], target_samples: int) -> List[int]:
    """Extend waveform to target duration by repeating the signal."""
    if len(samples) >= target_samples:
        return samples[:target_samples]

    extended = []
    while len(extended) < target_samples:
        remaining = target_samples - len(extended)
        extended.extend(samples[:remaining])

    return extended[:target_samples]


def encode_muse_waveform(samples: List[int]) -> str:
    """Encode waveform to GE MUSE Base64 format with line breaks."""
    raw_bytes = struct.pack(f'<{len(samples)}h', *samples)
    b64_data = base64.b64encode(raw_bytes).decode('ascii')
    lines = [b64_data[i:i+68] for i in range(0, len(b64_data), 68)]
    return '\n'.join(lines)


def calculate_crc32(samples: List[int]) -> int:
    """Calculate CRC32 checksum for waveform data."""
    raw_bytes = struct.pack(f'<{len(samples)}h', *samples)
    return zlib.crc32(raw_bytes) & 0xffffffff


def extract_philips_patient_info(root: ET.Element, ns: Dict[str, str]) -> Dict[str, str]:
    """Extract patient information from Philips XML."""
    info = {
        'patient_id': '',
        'firstname': '',
        'lastname': '',
        'birthdate': '',
        'sex': '',
        'acquisition_date': '',
        'acquisition_time': '',
    }

    try:
        patient = root.find('.//patient', ns) or root.find('.//patient')
        if patient is not None:
            pid = patient.find('.//patientid', ns) or patient.find('.//patientid')
            if pid is not None and pid.text:
                info['patient_id'] = pid.text.strip()

            name = patient.find('.//name', ns) or patient.find('.//name')
            if name is not None:
                fn = name.find('firstname', ns) or name.find('firstname')
                ln = name.find('lastname', ns) or name.find('lastname')
                if fn is not None and fn.text:
                    info['firstname'] = fn.text.strip()
                if ln is not None and ln.text:
                    info['lastname'] = ln.text.strip()

            age = patient.find('.//age', ns) or patient.find('.//age')
            if age is not None:
                dob = age.find('dateofbirth', ns) or age.find('dateofbirth')
                if dob is not None and dob.text:
                    info['birthdate'] = dob.text.strip()

            sex = patient.find('.//sex', ns) or patient.find('.//sex')
            if sex is not None and sex.text:
                info['sex'] = sex.text.strip()

        acq = root.find('.//dataacquisition', ns) or root.find('.//dataacquisition')
        if acq is not None:
            info['acquisition_date'] = acq.get('date', '')
            info['acquisition_time'] = acq.get('time', '')

    except Exception as e:
        logger.warning(f"Error extracting patient info: {e}")

    return info


def extract_philips_waveforms(xml_path: str) -> Tuple[Dict[str, List[int]], Dict[str, str], int]:
    """
    Extract waveform data from Philips PageWriter XML.

    Returns:
        Tuple of (lead_data dict, patient_info dict, sampling_rate)
    """
    lead_data = {}
    patient_info = {}
    sampling_rate = 500

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        ns = {}
        if root.tag.startswith('{'):
            ns_uri = root.tag.split('}')[0] + '}'
            ns = {'': ns_uri[1:-1]}

        patient_info = extract_philips_patient_info(root, ns)

        # Find repbeats section
        repbeats = None
        for elem in root.iter():
            if 'repbeats' in elem.tag.lower():
                repbeats = elem
                break

        if repbeats is None:
            logger.warning("No repbeats section found in Philips XML")
            return lead_data, patient_info, sampling_rate

        sps = repbeats.get('samplespersec')
        if sps:
            sampling_rate = int(sps)

        for repbeat in repbeats.iter():
            if 'repbeat' in repbeat.tag.lower():
                lead_name = repbeat.get('leadname', '')
                if not lead_name:
                    continue

                waveform = None
                for child in repbeat:
                    if 'waveform' in child.tag.lower():
                        waveform = child
                        break

                if waveform is not None and waveform.text:
                    samples = decode_philips_waveform(waveform.text)
                    if samples:
                        lead_data[lead_name] = samples
                        logger.debug(f"Extracted {len(samples)} samples for lead {lead_name}")

        logger.info(f"Extracted {len(lead_data)} leads from Philips XML")

    except Exception as e:
        logger.error(f"Error extracting Philips waveforms: {e}")

    return lead_data, patient_info, sampling_rate


def create_muse_xml(lead_data: Dict[str, List[int]], patient_info: Dict[str, str],
                    original_rate: int = 1000, target_rate: int = 500,
                    target_duration_ms: int = 10000) -> str:
    """
    Create GE MUSE RestingECG format XML from extracted lead data.

    Args:
        lead_data: Dictionary mapping lead names to sample lists
        patient_info: Patient information dictionary
        original_rate: Original sampling rate from Philips
        target_rate: Target sampling rate (Hz) - GE MUSE uses 500
        target_duration_ms: Target duration in milliseconds

    Returns:
        GE MUSE RestingECG XML string
    """
    target_samples = (target_rate * target_duration_ms) // 1000

    # Process waveforms
    processed_leads = {}
    for lead_name in LEAD_ORDER_8:
        if lead_name in lead_data:
            samples = lead_data[lead_name]
            resampled = resample_waveform(samples, original_rate, target_rate)
            extended = extend_to_duration(resampled, target_samples)
            processed_leads[lead_name] = extended

    # Build GE MUSE RestingECG XML
    xml_parts = []
    xml_parts.append('<?xml version="1.0" encoding="utf-8"?>')
    xml_parts.append('<!DOCTYPE RestingECG SYSTEM "restecg.dtd">')
    xml_parts.append('<RestingECG>')

    # MuseInfo section
    xml_parts.append('   <MuseInfo>')
    xml_parts.append('      <MuseVersion>10.1.0.0</MuseVersion>')
    xml_parts.append('   </MuseInfo>')

    # PatientDemographics
    xml_parts.append('   <PatientDemographics>')
    xml_parts.append(f'      <PatientID>{patient_info.get("patient_id", "")}</PatientID>')
    if patient_info.get('lastname') or patient_info.get('firstname'):
        xml_parts.append(f'      <PatientLastName>{patient_info.get("lastname", "")}</PatientLastName>')
        xml_parts.append(f'      <PatientFirstName>{patient_info.get("firstname", "")}</PatientFirstName>')
    if patient_info.get('birthdate'):
        xml_parts.append(f'      <DateofBirth>{patient_info.get("birthdate")}</DateofBirth>')
    if patient_info.get('sex'):
        gender = 'MALE' if patient_info.get('sex', '').lower().startswith('m') else 'FEMALE'
        xml_parts.append(f'      <Gender>{gender}</Gender>')
    xml_parts.append('   </PatientDemographics>')

    # TestDemographics
    acq_date = patient_info.get('acquisition_date', datetime.now().strftime('%m-%d-%Y'))
    acq_time = patient_info.get('acquisition_time', datetime.now().strftime('%H:%M:%S'))
    xml_parts.append('   <TestDemographics>')
    xml_parts.append(f'      <AcquisitionDate>{acq_date}</AcquisitionDate>')
    xml_parts.append(f'      <AcquisitionTime>{acq_time}</AcquisitionTime>')
    xml_parts.append('   </TestDemographics>')

    # RestingECGMeasurements (minimal)
    xml_parts.append('   <RestingECGMeasurements>')
    xml_parts.append('      <VentricularRate>0</VentricularRate>')
    xml_parts.append('      <QRSDuration>0</QRSDuration>')
    xml_parts.append('      <QTInterval>0</QTInterval>')
    xml_parts.append(f'      <ECGSampleBase>{target_rate}</ECGSampleBase>')
    xml_parts.append('      <ECGSampleExponent>0</ECGSampleExponent>')
    xml_parts.append('   </RestingECGMeasurements>')

    # Waveform section (Rhythm - 10 second strip)
    byte_count = target_samples * 2

    xml_parts.append('   <Waveform>')
    xml_parts.append('      <WaveformType>Rhythm</WaveformType>')
    xml_parts.append('      <WaveformStartTime>0</WaveformStartTime>')
    xml_parts.append(f'      <NumberofLeads>{len(processed_leads)}</NumberofLeads>')
    xml_parts.append('      <SampleType>CONTINUOUS_SAMPLES</SampleType>')
    xml_parts.append(f'      <SampleBase>{target_rate}</SampleBase>')
    xml_parts.append('      <SampleExponent>0</SampleExponent>')
    xml_parts.append('      <HighPassFilter>5</HighPassFilter>')
    xml_parts.append('      <LowPassFilter>150</LowPassFilter>')
    xml_parts.append('      <ACFilter>60</ACFilter>')

    # Add lead data
    for lead_name in LEAD_ORDER_8:
        if lead_name in processed_leads:
            samples = processed_leads[lead_name]
            b64_data = encode_muse_waveform(samples)
            crc = calculate_crc32(samples)

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
    xml_parts.append('</RestingECG>')

    return '\n'.join(xml_parts)


def convert_philips_to_muse(input_path: str, output_path: Optional[str] = None) -> Tuple[bool, str]:
    """
    Convert Philips PageWriter TC XML to GE MUSE RestingECG format.

    Args:
        input_path: Path to Philips XML file
        output_path: Path for output file (optional)

    Returns:
        Tuple of (success, message_or_output_path)
    """
    try:
        if not detect_philips_pagewriter(input_path):
            return False, "Not a Philips PageWriter format"

        logger.info(f"Converting Philips file: {input_path}")

        lead_data, patient_info, sampling_rate = extract_philips_waveforms(input_path)

        if not lead_data:
            return False, "No waveform data could be extracted"

        if len(lead_data) < 8:
            return False, f"Insufficient leads extracted: {len(lead_data)}/8 minimum"

        muse_xml = create_muse_xml(lead_data, patient_info, original_rate=sampling_rate)

        if output_path is None:
            output_path = input_path

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(muse_xml)

        logger.info(f"Successfully converted to GE MUSE format: {output_path}")
        return True, output_path

    except Exception as e:
        logger.error(f"Error converting Philips to MUSE: {e}")
        return False, str(e)


def auto_convert_if_philips(xml_path: str) -> Tuple[bool, str, str]:
    """
    Automatically detect and convert Philips files to GE MUSE format.

    Args:
        xml_path: Path to XML file

    Returns:
        Tuple of (was_converted, format_detected, message)
    """
    try:
        if not detect_philips_pagewriter(xml_path):
            return False, "other", "Not a Philips file"

        success, result = convert_philips_to_muse(xml_path)

        if success:
            return True, "philips_pagewriter", "Converted Philips PageWriter to GE MUSE format"
        else:
            return False, "philips_pagewriter", f"Conversion failed: {result}"

    except Exception as e:
        return False, "unknown", f"Error: {str(e)}"
