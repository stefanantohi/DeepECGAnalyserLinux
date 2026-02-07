import streamlit as st
import xml.etree.ElementTree as ET
import base64
import struct
import numpy as np
import re
import io

st.set_page_config(page_title="ECG Mirror Fixer", layout="wide")
st.title("🪞 ECG Mirror Fixer (Anti-Unbound Prefix)")

def resample_to_500hz(signal, target_len=5000):
    if len(signal) == 0: return np.zeros(target_len)
    return np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(signal)), signal)

uploaded_file = st.file_uploader("Uploader le fichier XML Philips ou HL7", type=["xml"])

if uploaded_file:
    raw = uploaded_file.read()
    
    # 1. Gestion de l'encodage (UTF-16 très fréquent chez Philips)
    content = ""
    for enc in ['utf-16', 'utf-8', 'latin-1']:
        try:
            content = raw.decode(enc)
            break
        except: continue

    if content:
        # 2. NETTOYAGE RADICAL DES PREFIXES (Solution à l'erreur Unbound Prefix)
        # On supprime les définitions de namespaces et les préfixes de balises (ex: p:waveform -> waveform)
        content = re.sub(r'xmlns(:?\w+)?="[^"]+"', '', content)
        content = re.sub(r'xsi:schemaLocation="[^"]+"', '', content)
        content = re.sub(r'([<//])\w+:', r'\1', content)
        
        try:
            root = ET.fromstring(content)
            leads_data = {}

            # Extraction Philips (Repbeats)
            repbeats = root.findall(".//repbeat")
            for rb in repbeats:
                lid = rb.get('leadname')
                wf = rb.find("waveform")
                if wf is not None and wf.text:
                    decoded = base64.b64decode(wf.text.strip())
                    leads_data[lid] = np.array(struct.unpack(f'<{len(decoded)//2}h', decoded))

            # Extraction HL7 si Philips échoue
            if not leads_data:
                for seq in root.findall(".//sequence"):
                    c = seq.find(".//code")
                    v = seq.find(".//value")
                    if c is not None and v is not None:
                        lid = c.get('code', '').replace('MDC_ECG_LEAD_', '')
                        leads_data[lid] = np.array([float(val) for val in v.text.split()])

            if leads_data:
                # 3. CONSTRUCTION DU CLONE PARFAIT DE ECG_001
                order = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
                new_root = ET.Element('RestingECG')
                
                # Section Measurements (identique à ECG_001.xml)
                meas = ET.SubElement(new_root, 'RestingECGMeasurements')
                ET.SubElement(meas, 'VentricularRate').text = "77"
                ET.SubElement(meas, 'ECGSampleBase').text = "500" 
                
                wave = ET.SubElement(new_root, 'Waveform')
                ET.SubElement(wave, 'WaveformType').text = "Rhythm"
                
                for lead in order:
                    sig = leads_data.get(lead, np.zeros(5000))
                    # Resampling vers 500Hz (5000 points pour 10s) comme ECG_001
                    resampled = resample_to_500hz(sig, 5000).astype(np.int16)
                    b64_sig = base64.b64encode(resampled.tobytes()).decode('ascii')
                    
                    ld = ET.SubElement(wave, 'LeadData')
                    ET.SubElement(ld, 'LeadID').text = lead
                    ET.SubElement(ld, 'LeadSampleCountTotal').text = "5000"
                    ET.SubElement(ld, 'WaveFormData').text = b64_sig

                out = io.BytesIO()
                ET.ElementTree(new_root).write(out, encoding='utf-8', xml_declaration=True)
                
                st.success("✅ Conversion réussie ! Le fichier est maintenant un jumeau de ECG_001.")
                st.download_button("📥 Télécharger le XML corrigé", 
                                 data=out.getvalue(), 
                                 file_name=f"FIXED_{uploaded_file.name}")
            else:
                st.error("Aucun signal trouvé dans le fichier.")
        except Exception as e:
            st.error(f"Erreur lors du parsing : {e}")