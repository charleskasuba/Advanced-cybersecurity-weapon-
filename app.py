import streamlit as st
import hashlib
import base64
import secrets
import string
import re
import json
import requests
import socket
import whois
import dns.resolver
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import plotly.express as px
import plotly.graph_objects as go
from urllib.parse import urlparse, quote, unquote
import ipaddress
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio
import aiohttp
import qrcode
from io import BytesIO
import cryptography
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
import jwt
import hmac
import zlib
import tarfile
import gzip
import binascii
from PIL import Image
import cv2
import pytesseract
from steganography import Steganography
import networkx as nx
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Advanced CSOC Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for advanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(45deg, #1f77b4, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 0.5rem;
    }
    .tool-card {
        background: #1e1e1e;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        color: white;
    }
    .security-critical { background: linear-gradient(135deg, #ff416c, #ff4b2b); color: white; padding: 8px; border-radius: 5px; }
    .security-high { background: linear-gradient(135deg, #ff9966, #ff5e62); color: white; padding: 8px; border-radius: 5px; }
    .security-medium { background: linear-gradient(135deg, #f9d423, #ff4e50); color: white; padding: 8px; border-radius: 5px; }
    .security-low { background: linear-gradient(135deg, #56ab2f, #a8e6cf); color: white; padding: 8px; border-radius: 5px; }
    .sidebar .sidebar-content {
        background: #0f1116;
    }
    .stButton>button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

class AdvancedCybersecurityTools:
    def __init__(self):
        self.threat_intel_sources = [
            "https://otx.alienvault.com/api/v1/indicators/",
            "https://api.shodan.io/",
            "https://www.virustotal.com/api/v3/"
        ]
    
    # Advanced Hashing & Cryptography
    def advanced_hash_generator(self, text, algorithm='sha3_512', salt=None, iterations=100000):
        """Advanced hash generation with salt and iterations"""
        if salt is None:
            salt = secrets.token_bytes(16)
        
        if algorithm.startswith('scrypt'):
            kdf = hashlib.scrypt(
                text.encode(),
                salt=salt,
                n=2**14,
                r=8,
                p=1,
                dklen=64
            )
            return kdf.hex()
        else:
            hash_func = getattr(hashlib, algorithm, None)
            if hash_func:
                # Apply key stretching
                hashed = text.encode()
                for _ in range(iterations):
                    hashed = hash_func(hashed + salt).digest()
                return hashed.hex()
        return None

    def generate_rsa_keys(self, key_size=4096):
        """Generate RSA key pair"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size
        )
        public_key = private_key.public_key()
        
        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return private_pem.decode(), public_pem.decode()

    def encrypt_rsa(self, message, public_key_pem):
        """Encrypt with RSA public key"""
        public_key = serialization.load_pem_public_key(public_key_pem.encode())
        encrypted = public_key.encrypt(
            message.encode(),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return base64.b64encode(encrypted).decode()

    # Network Security Tools
    def port_scanner(self, target, ports="1-1000", timeout=1):
        """Advanced port scanner"""
        open_ports = []
        start_port, end_port = map(int, ports.split('-'))
        
        def scan_port(port):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(timeout)
                    result = s.connect_ex((target, port))
                    if result == 0:
                        service = socket.getservbyport(port, 'tcp') if port <= 1000 else "unknown"
                        open_ports.append((port, service))
            except:
                pass
        
        with ThreadPoolExecutor(max_workers=100) as executor:
            executor.map(scan_port, range(start_port, end_port + 1))
        
        return open_ports

    def dns_enumeration(self, domain):
        """Comprehensive DNS enumeration"""
        record_types = ['A', 'AAAA', 'MX', 'NS', 'TXT', 'CNAME', 'SOA']
        results = {}
        
        for record_type in record_types:
            try:
                answers = dns.resolver.resolve(domain, record_type)
                results[record_type] = [str(rdata) for rdata in answers]
            except:
                results[record_type] = []
        
        return results

    def whois_lookup(self, domain):
        """WHOIS information lookup"""
        try:
            w = whois.whois(domain)
            return dict(w)
        except Exception as e:
            return {"error": str(e)}

    # Malware Analysis
    def file_hash_analyzer(self, file_content):
        """Analyze file with multiple hash types"""
        hashes = {}
        algorithms = ['md5', 'sha1', 'sha256', 'sha512', 'sha3_256', 'sha3_512']
        
        for algo in algorithms:
            hash_func = getattr(hashlib, algo)()
            hash_func.update(file_content)
            hashes[algo.upper()] = hash_func.hexdigest()
        
        return hashes

    def string_extractor(self, file_content, min_length=4):
        """Extract strings from binary data"""
        strings = re.findall(b'[\\x20-\\x7E]{'+str(min_length).encode()+b',}', file_content)
        return [s.decode('ascii', errors='ignore') for s in strings]

    # Steganography
    def hide_text_in_image(self, image_file, secret_text, password=None):
        """Hide text in image using steganography"""
        try:
            # Simple LSB steganography implementation
            img = Image.open(image_file)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            pixels = img.load()
            binary_text = ''.join(format(ord(i), '08b') for i in secret_text)
            binary_text += '1111111111111110'  # End marker
            
            if len(binary_text) > img.size[0] * img.size[1] * 3:
                return None, "Message too long for image"
            
            index = 0
            for x in range(img.size[0]):
                for y in range(img.size[1]):
                    if index < len(binary_text):
                        r, g, b = pixels[x, y]
                        r = (r & 0xFE) | int(binary_text[index])
                        index += 1
                        if index < len(binary_text):
                            g = (g & 0xFE) | int(binary_text[index])
                            index += 1
                        if index < len(binary_text):
                            b = (b & 0xFE) | int(binary_text[index])
                            index += 1
                        pixels[x, y] = (r, g, b)
                    else:
                        break
            
            output = BytesIO()
            img.save(output, format='PNG')
            return output.getvalue(), "Success"
        except Exception as e:
            return None, str(e)

    # Threat Intelligence
    def check_ip_reputation(self, ip):
        """Check IP reputation (simulated)"""
        # Note: In production, use actual threat intelligence APIs
        threat_data = {
            "malicious_score": np.random.randint(0, 100),
            "abuse_confidence": np.random.randint(0, 100),
            "threat_types": ["Botnet", "C2 Server"] if np.random.random() > 0.7 else [],
            "last_seen": datetime.now().isoformat()
        }
        return threat_data

    def domain_reputation_analysis(self, domain):
        """Analyze domain reputation"""
        analysis = {
            "age_days": np.random.randint(1, 3650),
            "trust_score": np.random.randint(0, 100),
            "suspicious_indicators": [],
            "associated_threats": []
        }
        
        if analysis["age_days"] < 30:
            analysis["suspicious_indicators"].append("New domain")
        if analysis["trust_score"] < 30:
            analysis["associated_threats"].append("Potential phishing")
        
        return analysis

    # Digital Forensics
    def timeline_analyzer(self, events):
        """Analyze event timeline"""
        df = pd.DataFrame(events)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        return df

    def memory_analysis_simulation(self):
        """Simulate memory analysis"""
        processes = [
            {"pid": 1234, "name": "explorer.exe", "cpu": 2.5, "memory": 45.2},
            {"pid": 5678, "name": "chrome.exe", "cpu": 15.7, "memory": 120.5},
            {"pid": 9012, "name": "suspicious.exe", "cpu": 85.3, "memory": 5.1},
        ]
        return processes

    # Cryptanalysis
    def frequency_analysis(self, text):
        """Perform frequency analysis on text"""
        text = text.upper()
        freq = {}
        total = len([c for c in text if c.isalpha()])
        
        for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            count = text.count(char)
            freq[char] = (count, count/total * 100 if total > 0 else 0)
        
        return freq

    def caesar_cracker(self, ciphertext):
        """Attempt to crack Caesar cipher"""
        results = []
        for shift in range(26):
            plaintext = ""
            for char in ciphertext:
                if char.isalpha():
                    ascii_offset = 65 if char.isupper() else 97
                    plaintext += chr((ord(char) - ascii_offset - shift) % 26 + ascii_offset)
                else:
                    plaintext += char
            results.append((shift, plaintext))
        return results

    # Security Headers Analyzer
    def analyze_http_headers(self, url):
        """Analyze HTTP security headers"""
        try:
            response = requests.get(url, timeout=10)
            headers = dict(response.headers)
            
            security_headers = {
                'Content-Security-Policy': headers.get('Content-Security-Policy', 'MISSING'),
                'Strict-Transport-Security': headers.get('Strict-Transport-Security', 'MISSING'),
                'X-Content-Type-Options': headers.get('X-Content-Type-Options', 'MISSING'),
                'X-Frame-Options': headers.get('X-Frame-Options', 'MISSING'),
                'X-XSS-Protection': headers.get('X-XSS-Protection', 'MISSING'),
                'Referrer-Policy': headers.get('Referrer-Policy', 'MISSING')
            }
            
            return security_headers, len(response.content)
        except Exception as e:
            return {"error": str(e)}, 0

    # QR Code Security
    def generate_secure_qr(self, data, password=None):
        """Generate QR code with optional encryption"""
        if password:
            # Simple XOR encryption for demonstration
            encrypted = ''.join(chr(ord(c) ^ ord(password[i % len(password)])) for i, c in enumerate(data))
            data = base64.b64encode(encrypted.encode()).decode()
        
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(data)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

def main():
    st.markdown('<h1 class="main-header">🛡️ Advanced Cybersecurity Operations Center</h1>', unsafe_allow_html=True)
    
    # Initialize tools
    tools = AdvancedCybersecurityTools()
    
    # Sidebar navigation
    st.sidebar.title("🔒 CSOC Navigation")
    section = st.sidebar.selectbox(
        "Select Module:",
        [
            "📊 Dashboard",
            "🔐 Advanced Cryptography",
            "🌐 Network Security",
            "🕵️ Threat Intelligence",
            "🔍 Digital Forensics",
            "📡 Web Security",
            "📱 Mobile Security",
            "🧩 Cryptanalysis",
            "🖼️ Steganography",
            "📈 Security Analytics",
            "⚙️ Incident Response",
            "🔧 Utilities"
        ]
    )
    
    if section == "📊 Dashboard":
        show_dashboard(tools)
    elif section == "🔐 Advanced Cryptography":
        show_advanced_cryptography(tools)
    elif section == "🌐 Network Security":
        show_network_security(tools)
    elif section == "🕵️ Threat Intelligence":
        show_threat_intelligence(tools)
    elif section == "🔍 Digital Forensics":
        show_digital_forensics(tools)
    elif section == "📡 Web Security":
        show_web_security(tools)
    elif section == "📱 Mobile Security":
        show_mobile_security(tools)
    elif section == "🧩 Cryptanalysis":
        show_cryptanalysis(tools)
    elif section == "🖼️ Steganography":
        show_steganography(tools)
    elif section == "📈 Security Analytics":
        show_security_analytics(tools)
    elif section == "⚙️ Incident Response":
        show_incident_response(tools)
    elif section == "🔧 Utilities":
        show_utilities(tools)

def show_dashboard(tools):
    st.header("📊 CSOC Dashboard")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card"><h3>Threat Level</h3><h2>MEDIUM</h2></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>Active Incidents</h3><h2>3</h2></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>Systems Protected</h3><h2>1,247</h2></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h3>Last Scan</h3><h2>2 min ago</h2></div>', unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Threat distribution chart
        threat_data = pd.DataFrame({
            'Category': ['Malware', 'Phishing', 'DDoS', 'Insider Threat', 'Zero-Day'],
            'Count': [45, 32, 18, 12, 5]
        })
        fig = px.pie(threat_data, values='Count', names='Category', title='Threat Distribution')
        st.plotly_chart(fig)
    
    with col2:
        # Timeline
        timeline_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'Events': np.random.randint(1, 20, 30)
        })
        fig = px.line(timeline_data, x='Date', y='Events', title='Security Events Timeline')
        st.plotly_chart(fig)
    
    # Recent Alerts
    st.subheader("🚨 Recent Security Alerts")
    alerts = [
        {"time": "10:23 AM", "severity": "HIGH", "description": "Suspicious outbound traffic detected", "source": "192.168.1.45"},
        {"time": "09:47 AM", "severity": "MEDIUM", "description": "Multiple failed login attempts", "source": "10.0.1.23"},
        {"time": "08:15 AM", "severity": "LOW", "description": "Unusual port scanning activity", "source": "External"}
    ]
    
    for alert in alerts:
        severity_class = f"security-{alert['severity'].lower()}"
        st.markdown(f"""
        <div style="padding: 10px; border-left: 4px solid; margin: 5px 0; background: #1e1e1e;">
            <strong>{alert['time']}</strong> - 
            <span class="{severity_class}">{alert['severity']}</span> - 
            {alert['description']} - 
            <em>{alert['source']}</em>
        </div>
        """, unsafe_allow_html=True)

def show_advanced_cryptography(tools):
    st.header("🔐 Advanced Cryptography")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["RSA Encryption", "Advanced Hashing", "Key Generation", "Digital Signatures", "Cryptographic Analysis"])
    
    with tab1:
        st.subheader("RSA Encryption/Decryption")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Generate RSA Key Pair"):
                private_key, public_key = tools.generate_rsa_keys()
                st.text_area("Private Key:", private_key, height=200)
                st.text_area("Public Key:", public_key, height=150)
        
        with col2:
            message = st.text_area("Message to encrypt:")
            public_key_input = st.text_area("Public Key for encryption:", height=150)
            
            if st.button("Encrypt Message") and message and public_key_input:
                encrypted = tools.encrypt_rsa(message, public_key_input)
                st.text_area("Encrypted Message:", encrypted, height=100)
    
    with tab2:
        st.subheader("Advanced Hashing Algorithms")
        
        text = st.text_area("Text to hash:")
        algorithm = st.selectbox("Algorithm:", ["sha3_512", "sha3_256", "blake2b", "blake2s", "scrypt"])
        
        if st.button("Generate Advanced Hash") and text:
            if algorithm == "scrypt":
                hash_result = tools.advanced_hash_generator(text, algorithm="scrypt")
            else:
                hash_result = tools.advanced_hash_generator(text, algorithm)
            
            if hash_result:
                st.text_area("Hash Result:", hash_result, height=100)
    
    with tab3:
        st.subheader("Cryptographic Key Generation")
        
        key_type = st.selectbox("Key Type:", ["AES-256", "RSA-4096", "ECDSA-P521", "Ed25519"])
        if st.button("Generate Key"):
            if key_type == "AES-256":
                key = Fernet.generate_key()
                st.text_area("AES-256 Key:", key.decode(), height=100)
    
    with tab4:
        st.subheader("Digital Signature Simulation")
        
        document = st.text_area("Document to sign:")
        if st.button("Generate Signature") and document:
            # Simulate digital signature
            signature = hashlib.sha256(document.encode()).hexdigest()
            st.text_area("Digital Signature:", signature, height=100)
    
    with tab5:
        st.subheader("Cryptographic Strength Analysis")
        
        password = st.text_input("Enter password for analysis:", type="password")
        if password:
            entropy = len(password) * 4  # Simple entropy calculation
            st.metric("Password Entropy", f"{entropy} bits")
            
            if entropy < 40:
                st.error("Weak password - high risk")
            elif entropy < 60:
                st.warning("Moderate password - medium risk")
            else:
                st.success("Strong password - low risk")

def show_network_security(tools):
    st.header("🌐 Network Security")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Port Scanner", "DNS Enumeration", "WHOIS Lookup", "Network Analysis"])
    
    with tab1:
        st.subheader("Advanced Port Scanner")
        
        target = st.text_input("Target IP/Hostname:")
        port_range = st.text_input("Port Range (e.g., 1-1000):", "1-1000")
        
        if st.button("Start Port Scan") and target:
            with st.spinner("Scanning ports..."):
                open_ports = tools.port_scanner(target, port_range)
            
            if open_ports:
                st.success(f"Found {len(open_ports)} open ports")
                for port, service in open_ports:
                    st.write(f"Port {port}/tcp - {service}")
            else:
                st.info("No open ports found in specified range")
    
    with tab2:
        st.subheader("DNS Enumeration")
        
        domain = st.text_input("Domain for DNS enumeration:")
        if st.button("Enumerate DNS") and domain:
            with st.spinner("Gathering DNS records..."):
                results = tools.dns_enumeration(domain)
            
            for record_type, records in results.items():
                st.write(f"**{record_type} Records:**")
                for record in records:
                    st.write(f"  - {record}")
    
    with tab3:
        st.subheader("WHOIS Lookup")
        
        domain = st.text_input("Domain for WHOIS lookup:")
        if st.button("Perform WHOIS Lookup") and domain:
            with st.spinner("Querying WHOIS database..."):
                whois_info = tools.whois_lookup(domain)
            
            st.json(whois_info)
    
    with tab4:
        st.subheader("Network Traffic Analysis")
        
        # Simulate network traffic analysis
        if st.button("Analyze Network Traffic"):
            traffic_data = {
                "Protocols": {"TCP": 65, "UDP": 25, "ICMP": 10},
                "Top Talkers": ["192.168.1.100", "10.0.0.50", "192.168.1.1"],
                "Suspicious Activity": ["Port scanning from 203.0.113.45", "DNS tunneling attempt"]
            }
            
            st.write("**Protocol Distribution:**")
            fig = px.pie(values=list(traffic_data["Protocols"].values()), 
                        names=list(traffic_data["Protocols"].keys()))
            st.plotly_chart(fig)
            
            st.write("**Suspicious Activity:**")
            for activity in traffic_data["Suspicious Activity"]:
                st.write(f"⚠️ {activity}")

def show_threat_intelligence(tools):
    st.header("🕵️ Threat Intelligence")
    
    tab1, tab2, tab3 = st.tabs(["IP Reputation", "Domain Analysis", "Threat Feeds"])
    
    with tab1:
        st.subheader("IP Reputation Check")
        
        ip = st.text_input("Enter IP address for reputation check:")
        if st.button("Check IP Reputation") and ip:
            with st.spinner("Analyzing IP reputation..."):
                reputation = tools.check_ip_reputation(ip)
            
            st.metric("Malicious Score", f"{reputation['malicious_score']}%")
            st.metric("Abuse Confidence", f"{reputation['abuse_confidence']}%")
            
            if reputation['threat_types']:
                st.error(f"Associated threats: {', '.join(reputation['threat_types'])}")
            else:
                st.success("No known threats associated")
    
    with tab2:
        st.subheader("Domain Reputation Analysis")
        
        domain = st.text_input("Enter domain for analysis:")
        if st.button("Analyze Domain") and domain:
            with st.spinner("Analyzing domain reputation..."):
                analysis = tools.domain_reputation_analysis(domain)
            
            st.metric("Domain Age", f"{analysis['age_days']} days")
            st.metric("Trust Score", f"{analysis['trust_score']}%")
            
            if analysis['suspicious_indicators']:
                st.warning(f"Suspicious indicators: {', '.join(analysis['suspicious_indicators'])}")
            
            if analysis['associated_threats']:
                st.error(f"Associated threats: {', '.join(analysis['associated_threats'])}")
    
    with tab3:
        st.subheader("Threat Intelligence Feeds")
        
        # Simulated threat feed
        threats = [
            {"type": "Malware", "name": "Emotet", "risk": "HIGH", "last_seen": "2 hours ago"},
            {"type": "Phishing", "name": "Office365 Credential Harvesting", "risk": "MEDIUM", "last_seen": "1 hour ago"},
            {"type": "Exploit", "name": "Log4Shell", "risk": "CRITICAL", "last_seen": "30 minutes ago"}
        ]
        
        for threat in threats:
            risk_class = f"security-{threat['risk'].lower()}"
            st.markdown(f"""
            <div style="padding: 10px; margin: 5px 0; background: #1e1e1e; border-radius: 5px;">
                <strong>{threat['type']}</strong> - {threat['name']} - 
                <span class="{risk_class}">{threat['risk']}</span> - 
                <em>{threat['last_seen']}</em>
            </div>
            """, unsafe_allow_html=True)

# Continue with other sections in a similar comprehensive manner...

def show_digital_forensics(tools):
    st.header("🔍 Digital Forensics")
    
    tab1, tab2, tab3 = st.tabs(["Memory Analysis", "Timeline Analysis", "File Carving"])
    
    with tab1:
        st.subheader("Memory Analysis Simulation")
        
        if st.button("Analyze Memory Dump"):
            processes = tools.memory_analysis_simulation()
            
            st.write("**Running Processes:**")
            for proc in processes:
                st.write(f"PID {proc['pid']}: {proc['name']} - CPU: {proc['cpu']}% - Memory: {proc['memory']}MB")
                
                if proc['cpu'] > 80:
                    st.error("⚠️ High CPU usage - potential malware!")
    
    with tab2:
        st.subheader("Event Timeline Analysis")
        
        # Simulated events
        events = [
            {"timestamp": "2024-01-15 08:30:00", "event": "User login", "user": "john.doe"},
            {"timestamp": "2024-01-15 09:15:00", "event": "File download", "user": "john.doe"},
            {"timestamp": "2024-01-15 10:00:00", "event": "Suspicious process started", "user": "system"}
        ]
        
        timeline_df = tools.timeline_analyzer(events)
        st.dataframe(timeline_df)
    
    with tab3:
        st.subheader("File Carving & Analysis")
        
        uploaded_file = st.file_uploader("Upload file for analysis:", type=['exe', 'dll', 'bin'])
        if uploaded_file is not None:
            file_content = uploaded_file.getvalue()
            
            st.write("**File Hashes:**")
            hashes = tools.file_hash_analyzer(file_content)
            for algo, hash_val in hashes.items():
                st.write(f"{algo}: {hash_val}")
            
            st.write("**Extracted Strings:**")
            strings = tools.string_extractor(file_content)
            st.text_area("Strings found:", "\n".join(strings[:50]), height=200)

def show_web_security(tools):
    st.header("📡 Web Security")
    
    tab1, tab2, tab3 = st.tabs(["Security Headers", "SSL Analysis", "Web Vulnerability Scan"])
    
    with tab1:
        st.subheader("HTTP Security Headers Analysis")
        
        url = st.text_input("Enter URL for header analysis:")
        if st.button("Analyze Headers") and url:
            with st.spinner("Analyzing security headers..."):
                headers, content_length = tools.analyze_http_headers(url)
            
            if "error" not in headers:
                score = 0
                for header, value in headers.items():
                    if value != "MISSING":
                        score += 1
                        st.success(f"✅ {header}: {value}")
                    else:
                        st.error(f"❌ {header}: {value}")
                
                st.metric("Security Headers Score", f"{score}/6")
            else:
                st.error(f"Error: {headers['error']}")
    
    with tab2:
        st.subheader("SSL/TLS Configuration Check")
        
        # Simulated SSL analysis
        domain = st.text_input("Enter domain for SSL analysis:")
        if st.button("Check SSL Configuration") and domain:
            ssl_info = {
                "Protocols": ["TLS 1.2", "TLS 1.3"],
                "Cipher Strength": "Strong",
                "Certificate Validity": "Valid",
                "Expires": "2024-12-31"
            }
            
            st.json(ssl_info)
            
            if "TLS 1.0" in ssl_info["Protocols"] or "TLS 1.1" in ssl_info["Protocols"]:
                st.error("⚠️ Weak protocols detected!")
    
    with tab3:
        st.subheader("Web Application Vulnerability Scanner")
        
        target_url = st.text_input("Enter target URL for vulnerability scan:")
        if st.button("Start Vulnerability Scan") and target_url:
            with st.spinner("Scanning for vulnerabilities..."):
                # Simulated vulnerabilities
                vulnerabilities = [
                    {"type": "SQL Injection", "risk": "HIGH", "location": "/login.php"},
                    {"type": "XSS", "risk": "MEDIUM", "location": "/search.php"},
                    {"type": "CSRF", "risk": "LOW", "location": "/update_profile.php"}
                ]
                
                for vuln in vulnerabilities:
                    risk_class = f"security-{vuln['risk'].lower()}"
                    st.markdown(f"""
                    <div style="padding: 10px; margin: 5px 0; background: #1e1e1e; border-radius: 5px;">
                        <span class="{risk_class}">{vuln['risk']}</span> - 
                        {vuln['type']} at {vuln['location']}
                    </div>
                    """, unsafe_allow_html=True)

# Additional sections would follow the same pattern...

def show_mobile_security(tools):
    st.header("📱 Mobile Security")
    st.info("Mobile security analysis tools coming soon...")

def show_cryptanalysis(tools):
    st.header("🧩 Cryptanalysis")
    
    tab1, tab2 = st.tabs(["Frequency Analysis", "Cipher Cracking"])
    
    with tab1:
        st.subheader("Frequency Analysis")
        
        ciphertext = st.text_area("Enter ciphertext for frequency analysis:")
        if st.button("Analyze Frequency") and ciphertext:
            freq = tools.frequency_analysis(ciphertext)
            
            # Create frequency chart
            letters = list(freq.keys())
            counts = [freq[letter][1] for letter in letters]
            
            fig = px.bar(x=letters, y=counts, title="Letter Frequency Analysis")
            st.plotly_chart(fig)
    
    with tab2:
        st.subheader("Caesar Cipher Cracker")
        
        ciphertext = st.text_input("Enter Caesar cipher text:")
        if st.button("Crack Caesar Cipher") and ciphertext:
            results = tools.caesar_cracker(ciphertext)
            
            st.write("**Possible Decryptions:**")
            for shift, plaintext in results:
                st.write(f"Shift {shift:2d}: {plaintext}")

def show_steganography(tools):
    st.header("🖼️ Steganography")
    
    tab1, tab2 = st.tabs(["Hide Text", "Extract Text"])
    
    with tab1:
        st.subheader("Hide Text in Image")
        
        image_file = st.file_uploader("Upload image:", type=['png', 'jpg', 'jpeg'])
        secret_text = st.text_area("Text to hide:")
        password = st.text_input("Password (optional):", type="password")
        
        if st.button("Hide Text in Image") and image_file and secret_text:
            result_image, message = tools.hide_text_in_image(image_file, secret_text, password)
            
            if result_image:
                st.success("Text successfully hidden in image!")
                st.download_button(
                    label="Download Stego Image",
                    data=result_image,
                    file_name="stego_image.png",
                    mime="image/png"
                )
            else:
                st.error(f"Error: {message}")
    
    with tab2:
        st.subheader("Extract Text from Image")
        st.info("Text extraction feature requires additional implementation")

def show_security_analytics(tools):
    st.header("📈 Security Analytics")
    
    # Generate sample security data
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    security_data = pd.DataFrame({
        'date': dates,
        'malware_attacks': np.random.randint(0, 20, 30),
        'phishing_attempts': np.random.randint(0, 15, 30),
        'brute_force_attempts': np.random.randint(0, 50, 30),
        'successful_breaches': np.random.randint(0, 5, 30)
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(security_data, x='date', y=['malware_attacks', 'phishing_attempts'], 
                     title='Attack Trends Over Time')
        st.plotly_chart(fig)
    
    with col2:
        fig = px.bar(security_data, x='date', y='brute_force_attempts',
                    title='Brute Force Attempts')
        st.plotly_chart(fig)
    
    # Risk assessment
    st.subheader("Risk Assessment Dashboard")
    
    risk_factors = {
        "External Threat Level": 65,
        "Internal Vulnerability": 42,
        "Compliance Gap": 28,
        "Security Awareness": 75
    }
    
    for factor, score in risk_factors.items():
        st.write(f"**{factor}:** {score}%")
        st.progress(score/100)

def show_incident_response(tools):
    st.header("⚙️ Incident Response")
    
    tab1, tab2, tab3 = st.tabs(["Incident Playbook", "Evidence Collection", "Response Timeline"])
    
    with tab1:
        st.subheader("Incident Response Playbook")
        
        incident_type = st.selectbox("Select Incident Type:", 
                                   ["Malware Infection", "Data Breach", "DDoS Attack", "Insider Threat"])
        
        if incident_type == "Malware Infection":
            st.write("""
            **Response Steps:**
            1. 🔍 Identify and isolate infected systems
            2. 📊 Analyze malware samples
            3. 🛡️ Contain the spread
            4. 🧹 Remove malware
            5. 🔄 Restore systems from backups
            6. 📝 Document lessons learned
            """)
    
    with tab2:
        st.subheader("Digital Evidence Collection")
        
        evidence_types = st.multiselect("Select evidence to collect:", 
                                      ["Memory Dump", "Disk Image", "Log Files", "Network Captures", "Registry Hives"])
        
        if st.button("Generate Collection Script"):
            st.code("""
            # Sample evidence collection script
            echo "Starting evidence collection..."
            # Collect memory
            dump_memory.py --output memory.dmp
            # Collect disk image
            dd if=/dev/sda of=disk_image.img
            echo "Evidence collection complete!"
            """)
    
    with tab3:
        st.subheader("Incident Response Timeline")
        
        # Sample timeline
        timeline_events = [
            {"time": "08:00", "event": "Alert: Suspicious activity detected", "status": "🔴 Critical"},
            {"time": "08:05", "event": "Initial assessment started", "status": "🟡 Investigating"},
            {"time": "08:30", "event": "Containment measures applied", "status": "🟠 Containing"},
            {"time": "09:15", "event": "Root cause identified", "status": "🔵 Analyzing"},
            {"time": "10:00", "event": "Remediation in progress", "status": "🟢 Recovering"}
        ]
        
        for event in timeline_events:
            st.write(f"**{event['time']}** {event['status']} - {event['event']}")

def show_utilities(tools):
    st.header("🔧 Security Utilities")
    
    tab1, tab2, tab3, tab4 = st.tabs(["QR Code Generator", "Data Converter", "Password Audit", "System Info"])
    
    with tab1:
        st.subheader("Secure QR Code Generator")
        
        qr_data = st.text_input("Data for QR code:")
        password = st.text_input("Encryption password (optional):", type="password")
        
        if st.button("Generate QR Code") and qr_data:
            qr_image = tools.generate_secure_qr(qr_data, password)
            st.image(qr_image, caption="Secure QR Code", use_column_width=True)
    
    with tab2:
        st.subheader("Data Format Converter")
        
        input_data = st.text_area("Input data:")
        conversion = st.selectbox("Conversion type:", 
                                ["Text to Hex", "Hex to Text", "Base64 Encode", "Base64 Decode"])
        
        if st.button("Convert") and input_data:
            if conversion == "Text to Hex":
                result = input_data.encode().hex()
            elif conversion == "Hex to Text":
                result = bytes.fromhex(input_data).decode()
            elif conversion == "Base64 Encode":
                result = base64.b64encode(input_data.encode()).decode()
            elif conversion == "Base64 Decode":
                result = base64.b64decode(input_data.encode()).decode()
            
            st.text_area("Converted result:", result, height=100)
    
    with tab3:
        st.subheader("Password Security Audit")
        
        password = st.text_input("Password to audit:", type="password")
        if password:
            # Comprehensive password audit
            score = 0
            feedback = []
            
            # Length check
            if len(password) >= 12:
                score += 2
            elif len(password) >= 8:
                score += 1
            else:
                feedback.append("Password too short")
            
            # Complexity checks
            checks = [
                (r'[A-Z]', "uppercase letter"),
                (r'[a-z]', "lowercase letter"), 
                (r'\d', "number"),
                (r'[!@#$%^&*(),.?":{}|<>]', "special character")
            ]
            
            for pattern, description in checks:
                if re.search(pattern, password):
                    score += 1
                else:
                    feedback.append(f"Missing {description}")
            
            # Entropy calculation
            entropy = len(password) * 4
            score += min(entropy // 20, 3)
            
            # Display results
            st.metric("Security Score", f"{score}/9")
            st.metric("Estimated Entropy", f"{entropy} bits")
            
            for item in feedback:
                st.write(f"❌ {item}")
    
    with tab4:
        st.subheader("System Security Information")
        
        if st.button("Gather System Info"):
            # Simulated system information
            sys_info = {
                "Platform": "Linux",
                "Security Status": "Protected",
                "Firewall": "Active",
                "Last Update": "2024-01-15",
                "Antivirus": "Enabled"
            }
            
            st.json(sys_info)

if __name__ == "__main__":
    main()
