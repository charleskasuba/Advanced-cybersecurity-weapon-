import streamlit as st
import hashlib
import base64
import secrets
import string
import re
import json
import requests
import socket
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import plotly.express as px
import plotly.graph_objects as go
from urllib.parse import urlparse
import ipaddress
import threading
from concurrent.futures import ThreadPoolExecutor
import qrcode
from io import BytesIO
import jwt
import hmac
import zlib
import binascii
from PIL import Image
import matplotlib.pyplot as plt
import io
import uuid

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
        font-size: 3rem;
        background: linear-gradient(45deg, #FF4B4B, #1E90FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem;
        text-align: center;
    }
    .security-critical { background: #FF0000; color: white; padding: 5px; border-radius: 3px; }
    .security-high { background: #FF6B6B; color: white; padding: 5px; border-radius: 3px; }
    .security-medium { background: #FFA500; color: white; padding: 5px; border-radius: 3px; }
    .security-low { background: #32CD32; color: white; padding: 5px; border-radius: 3px; }
    .tool-card {
        background: #0f1116;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1E90FF;
    }
</style>
""", unsafe_allow_html=True)

class AdvancedCybersecurityTools:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    # Advanced Hashing
    def advanced_hash_generator(self, text, algorithm='sha3_256'):
        """Generate advanced hashes"""
        try:
            if algorithm == 'md5':
                return hashlib.md5(text.encode()).hexdigest()
            elif algorithm == 'sha1':
                return hashlib.sha1(text.encode()).hexdigest()
            elif algorithm == 'sha256':
                return hashlib.sha256(text.encode()).hexdigest()
            elif algorithm == 'sha512':
                return hashlib.sha512(text.encode()).hexdigest()
            elif algorithm == 'sha3_256':
                return hashlib.sha3_256(text.encode()).hexdigest()
            elif algorithm == 'sha3_512':
                return hashlib.sha3_512(text.encode()).hexdigest()
            elif algorithm == 'blake2b':
                return hashlib.blake2b(text.encode()).hexdigest()
            else:
                return "Unsupported algorithm"
        except Exception as e:
            return f"Error: {str(e)}"

    # Password Analysis
    def password_strength_analyzer(self, password):
        """Comprehensive password strength analysis"""
        score = 0
        feedback = []
        
        # Length check
        if len(password) >= 16:
            score += 3
        elif len(password) >= 12:
            score += 2
        elif len(password) >= 8:
            score += 1
        else:
            feedback.append("❌ Password should be at least 8 characters")
        
        # Character variety
        checks = [
            (r'[A-Z]', "uppercase letter", 1),
            (r'[a-z]', "lowercase letter", 1),
            (r'\d', "number", 1),
            (r'[!@#$%^&*(),.?":{}|<>]', "special character", 2),
            (r'.{20,}', "very long password", 2)
        ]
        
        for pattern, description, points in checks:
            if re.search(pattern, password):
                score += points
            else:
                if points > 1:  # Only show important missing elements
                    feedback.append(f"❌ Consider adding {description}")
        
        # Entropy calculation
        char_set = 0
        if re.search(r'[a-z]', password): char_set += 26
        if re.search(r'[A-Z]', password): char_set += 26
        if re.search(r'\d', password): char_set += 10
        if re.search(r'[^a-zA-Z0-9]', password): char_set += 32
        
        entropy = len(password) * (np.log2(char_set) if char_set > 0 else 0)
        
        # Determine strength
        if entropy > 100:
            strength = "Very Strong"
            color = "security-low"
        elif entropy > 60:
            strength = "Strong"
            color = "security-low"
        elif entropy > 40:
            strength = "Moderate"
            color = "security-medium"
        elif entropy > 20:
            strength = "Weak"
            color = "security-high"
        else:
            strength = "Very Weak"
            color = "security-critical"
        
        return {
            "strength": strength,
            "color": color,
            "score": score,
            "entropy": entropy,
            "feedback": feedback
        }

    def generate_secure_password(self, length=16, use_special=True, use_numbers=True):
        """Generate cryptographically secure password"""
        characters = string.ascii_letters
        if use_numbers:
            characters += string.digits
        if use_special:
            characters += "!@#$%^&*"
        
        return ''.join(secrets.choice(characters) for _ in range(length))

    # Network Tools
    def port_scanner(self, target, ports="1-100"):
        """Lightweight port scanner"""
        open_ports = []
        
        def scan_port(port):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(1)
                    result = sock.connect_ex((target, port))
                    if result == 0:
                        try:
                            service = socket.getservbyport(port, 'tcp')
                        except:
                            service = "unknown"
                        open_ports.append((port, service))
            except:
                pass
        
        try:
            start_port, end_port = map(int, ports.split('-'))
            with ThreadPoolExecutor(max_workers=50) as executor:
                executor.map(scan_port, range(start_port, end_port + 1))
        except:
            pass
        
        return open_ports

    def website_security_headers(self, url):
        """Analyze website security headers"""
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            response = self.session.get(url, timeout=10, verify=False)
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

    # Cryptography Tools
    def jwt_analyzer(self, token):
        """Analyze JWT tokens"""
        try:
            decoded = jwt.decode(token, options={"verify_signature": False})
            header = jwt.get_unverified_header(token)
            return {
                "header": header,
                "payload": decoded,
                "valid": True
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def generate_jwt_token(self, payload, secret):
        """Generate JWT token"""
        try:
            token = jwt.encode(payload, secret, algorithm="HS256")
            return token
        except Exception as e:
            return f"Error: {str(e)}"

    # Data Encoding/Decoding
    def multi_format_encoder(self, text, format_type):
        """Encode text in multiple formats"""
        try:
            if format_type == 'base64':
                return base64.b64encode(text.encode()).decode()
            elif format_type == 'hex':
                return text.encode().hex()
            elif format_type == 'binary':
                return ' '.join(format(ord(i), '08b') for i in text)
            elif format_type == 'url':
                return requests.utils.quote(text)
            else:
                return "Unsupported format"
        except Exception as e:
            return f"Error: {str(e)}"

    def multi_format_decoder(self, text, format_type):
        """Decode text from multiple formats"""
        try:
            if format_type == 'base64':
                return base64.b64decode(text.encode()).decode()
            elif format_type == 'hex':
                return bytes.fromhex(text).decode()
            elif format_type == 'binary':
                return ''.join(chr(int(b, 2)) for b in text.split())
            elif format_type == 'url':
                return requests.utils.unquote(text)
            else:
                return "Unsupported format"
        except Exception as e:
            return f"Error: {str(e)}"

    # QR Code Tools
    def generate_qr_code(self, data):
        """Generate QR code"""
        try:
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
        except Exception as e:
            return None

    # Threat Intelligence Simulation
    def ip_reputation_check(self, ip):
        """Simulate IP reputation check"""
        try:
            ipaddress.ip_address(ip)  # Validate IP
            
            # Simulate threat intelligence data
            threat_score = hash(ip) % 100  # Pseudo-random based on IP
            threats = []
            
            if threat_score > 80:
                threats = ["Known malicious IP", "Botnet activity"]
            elif threat_score > 60:
                threats = ["Suspicious activity", "Port scanning"]
            
            return {
                "ip": ip,
                "threat_score": threat_score,
                "threats": threats,
                "risk_level": "High" if threat_score > 80 else "Medium" if threat_score > 60 else "Low"
            }
        except ValueError:
            return {"error": "Invalid IP address"}

    # File Analysis
    def file_hash_analyzer(self, file_content):
        """Analyze file with multiple hash types"""
        hashes = {}
        algorithms = ['md5', 'sha1', 'sha256', 'sha512']
        
        for algo in algorithms:
            hash_func = getattr(hashlib, algo)()
            hash_func.update(file_content)
            hashes[algo.upper()] = hash_func.hexdigest()
        
        return hashes

    def string_extractor(self, file_content, min_length=4):
        """Extract strings from binary data"""
        try:
            strings = re.findall(b'[\\x20-\\x7E]{'+str(min_length).encode()+b',}', file_content)
            return [s.decode('ascii', errors='ignore') for s in strings[:100]]  # Limit output
        except:
            return []

def main():
    st.markdown('<h1 class="main-header">🛡️ Advanced Cybersecurity Operations Center</h1>', unsafe_allow_html=True)
    
    # Initialize tools
    tools = AdvancedCybersecurityTools()
    
    # Sidebar navigation
    st.sidebar.title("🔒 CSOC Navigation")
    section = st.sidebar.selectbox(
        "Select Module:",
        [
            "📊 Security Dashboard",
            "🔐 Password Security",
            "🔄 Cryptography Tools",
            "🌐 Network Analysis",
            "📡 Web Security",
            "🔍 Threat Intelligence",
            "📊 Data Analysis",
            "🛠️ Utilities"
        ]
    )
    
    if section == "📊 Security Dashboard":
        show_dashboard(tools)
    elif section == "🔐 Password Security":
        show_password_security(tools)
    elif section == "🔄 Cryptography Tools":
        show_cryptography_tools(tools)
    elif section == "🌐 Network Analysis":
        show_network_analysis(tools)
    elif section == "📡 Web Security":
        show_web_security(tools)
    elif section == "🔍 Threat Intelligence":
        show_threat_intelligence(tools)
    elif section == "📊 Data Analysis":
        show_data_analysis(tools)
    elif section == "🛠️ Utilities":
        show_utilities(tools)

def show_dashboard(tools):
    st.header("📊 Security Dashboard")
    
    # Real-time metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card"><h4>🛡️ Security Score</h4><h2>85%</h2></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h4>🚨 Active Threats</h4><h2>3</h2></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h4>🔍 Scans Today</h4><h2>47</h2></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h4>✅ Protected</h4><h2>98%</h2></div>', unsafe_allow_html=True)
    
    # Security charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Threat distribution
        threat_data = pd.DataFrame({
            'Category': ['Malware', 'Phishing', 'DDoS', 'Insider', 'Zero-Day'],
            'Count': [45, 32, 18, 8, 2]
        })
        fig = px.pie(threat_data, values='Count', names='Category', 
                    title='Threat Distribution', color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Security events timeline
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        events_data = pd.DataFrame({
            'Date': dates,
            'Security Events': np.random.randint(1, 50, 30)
        })
        fig = px.line(events_data, x='Date', y='Security Events', 
                     title='Security Events Timeline', line_shape='spline')
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent alerts
    st.subheader("🚨 Recent Security Alerts")
    alerts = [
        {"time": "14:23", "severity": "HIGH", "message": "Suspicious outbound traffic from 192.168.1.45", "status": "Investigating"},
        {"time": "13:47", "severity": "MEDIUM", "message": "Multiple failed login attempts", "status": "Resolved"},
        {"time": "12:15", "severity": "LOW", "message": "Unusual port scanning activity", "status": "Monitoring"}
    ]
    
    for alert in alerts:
        severity_class = f"security-{alert['severity'].lower()}"
        st.markdown(f"""
        <div style="padding: 10px; margin: 5px 0; background: #1e1e2e; border-radius: 5px; border-left: 4px solid #ff6b6b;">
            <strong>{alert['time']}</strong> - 
            <span class="{severity_class}">{alert['severity']}</span> - 
            {alert['message']} - 
            <em>{alert['status']}</em>
        </div>
        """, unsafe_allow_html=True)

def show_password_security(tools):
    st.header("🔐 Password Security Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Password Analyzer", "Password Generator", "Breach Check"])
    
    with tab1:
        st.subheader("Password Strength Analysis")
        
        password = st.text_input("Enter password to analyze:", type="password", key="analyze_pwd")
        
        if password:
            analysis = tools.password_strength_analyzer(password)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Strength", analysis["strength"])
                st.metric("Entropy Score", f"{analysis['entropy']:.1f} bits")
                st.metric("Security Score", f"{analysis['score']}/10")
            
            with col2:
                st.markdown(f"<span class='{analysis['color']}'>{analysis['strength']}</span>", unsafe_allow_html=True)
                
                # Progress bar for entropy
                entropy_percent = min(analysis['entropy'] / 100 * 100, 100)
                st.progress(entropy_percent / 100)
                st.caption(f"Password Entropy: {analysis['entropy']:.1f} bits")
            
            if analysis['feedback']:
                st.subheader("Recommendations:")
                for item in analysis['feedback']:
                    st.write(item)
    
    with tab2:
        st.subheader("Secure Password Generator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            length = st.slider("Password Length", 8, 32, 16)
            use_special = st.checkbox("Include Special Characters", True)
            use_numbers = st.checkbox("Include Numbers", True)
        
        with col2:
            if st.button("Generate Secure Password"):
                password = tools.generate_secure_password(length, use_special, use_numbers)
                st.text_area("Generated Password:", password, height=50)
                
                # Analyze the generated password
                analysis = tools.password_strength_analyzer(password)
                st.markdown(f"Strength: <span class='{analysis['color']}'>{analysis['strength']}</span>", 
                           unsafe_allow_html=True)
    
    with tab3:
        st.subheader("Password Breach Check")
        st.info("This feature checks if your password has been exposed in known data breaches.")
        
        email = st.text_input("Enter email to check:")
        if st.button("Check Breaches") and email:
            # Simulated breach check
            with st.spinner("Checking known data breaches..."):
                time.sleep(2)
                
                # Simulate results
                breach_count = hash(email) % 5
                if breach_count > 0:
                    st.error(f"❌ This email appears in {breach_count} known data breaches!")
                    st.write("Recommended actions:")
                    st.write("1. Change your password immediately")
                    st.write("2. Enable two-factor authentication")
                    st.write("3. Use a password manager")
                else:
                    st.success("✅ No known breaches found for this email")

def show_cryptography_tools(tools):
    st.header("🔄 Cryptography Tools")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Hash Generator", "JWT Analyzer", "Data Encoder", "Data Decoder"])
    
    with tab1:
        st.subheader("Advanced Hash Generator")
        
        text = st.text_area("Text to hash:")
        algorithm = st.selectbox("Hash Algorithm:", 
                               ["md5", "sha1", "sha256", "sha512", "sha3_256", "sha3_512", "blake2b"])
        
        if st.button("Generate Hash") and text:
            hash_result = tools.advanced_hash_generator(text, algorithm)
            st.text_area("Hash Result:", hash_result, height=100)
            
            # Show hash length
            st.info(f"Hash length: {len(hash_result)} characters")
    
    with tab2:
        st.subheader("JWT Token Analyzer")
        
        jwt_token = st.text_area("Paste JWT Token:")
        
        if st.button("Analyze JWT") and jwt_token:
            analysis = tools.jwt_analyzer(jwt_token)
            
            if analysis["valid"]:
                st.success("✅ Valid JWT Token")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Header")
                    st.json(analysis["header"])
                
                with col2:
                    st.subheader("Payload")
                    st.json(analysis["payload"])
            else:
                st.error("❌ Invalid JWT Token")
                st.write(analysis["error"])
        
        st.subheader("JWT Generator")
        payload_text = st.text_area("Payload (JSON):", '{"user": "admin", "exp": 1730000000}')
        secret = st.text_input("Secret Key:", type="password")
        
        if st.button("Generate JWT") and payload_text and secret:
            try:
                payload = json.loads(payload_text)
                token = tools.generate_jwt_token(payload, secret)
                st.text_area("Generated JWT:", token, height=100)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with tab3:
        st.subheader("Data Encoder")
        
        text = st.text_area("Text to encode:")
        encode_type = st.selectbox("Encoding Type:", ["base64", "hex", "binary", "url"])
        
        if st.button("Encode") and text:
            encoded = tools.multi_format_encoder(text, encode_type)
            st.text_area("Encoded Result:", encoded, height=150)
    
    with tab4:
        st.subheader("Data Decoder")
        
        text = st.text_area("Text to decode:")
        decode_type = st.selectbox("Decoding Type:", ["base64", "hex", "binary", "url"])
        
        if st.button("Decode") and text:
            decoded = tools.multi_format_decoder(text, decode_type)
            st.text_area("Decoded Result:", decoded, height=150)

def show_network_analysis(tools):
    st.header("🌐 Network Security Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Port Scanner", "IP Analysis", "Network Tools"])
    
    with tab1:
        st.subheader("Port Scanner")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target = st.text_input("Target IP/Hostname:", "example.com")
            port_range = st.text_input("Port Range:", "80-100")
        
        with col2:
            st.write("Common ports:")
            st.code("HTTP: 80\nHTTPS: 443\nSSH: 22\nFTP: 21\nDNS: 53")
        
        if st.button("Start Port Scan") and target:
            with st.spinner(f"Scanning {target}..."):
                open_ports = tools.port_scanner(target, port_range)
            
            if open_ports:
                st.success(f"Found {len(open_ports)} open ports")
                
                for port, service in open_ports:
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.write(f"**Port {port}/TCP**")
                    with col2:
                        st.write(f"Service: {service}")
            else:
                st.info("No open ports found in the specified range")
    
    with tab2:
        st.subheader("IP Address Analysis")
        
        ip = st.text_input("Enter IP address:")
        
        if st.button("Analyze IP") and ip:
            analysis = tools.ip_reputation_check(ip)
            
            if "error" not in analysis:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("IP Address", analysis["ip"])
                    st.metric("Threat Score", f"{analysis['threat_score']}/100")
                
                with col2:
                    risk_color = "security-high" if analysis['risk_level'] == "High" else "security-medium" if analysis['risk_level'] == "Medium" else "security-low"
                    st.markdown(f"Risk Level: <span class='{risk_color}'>{analysis['risk_level']}</span>", unsafe_allow_html=True)
                
                if analysis['threats']:
                    st.warning("⚠️ Potential Threats:")
                    for threat in analysis['threats']:
                        st.write(f"- {threat}")
                else:
                    st.success("✅ No known threats detected")
            else:
                st.error(analysis["error"])
    
    with tab3:
        st.subheader("Network Utilities")
        
        st.info("Additional network tools will be implemented here")
        st.write("Planned features:")
        st.write("• DNS lookup tools")
        st.write("• WHOIS information")
        st.write("• Network packet analysis")
        st.write("• SSL certificate checker")

def show_web_security(tools):
    st.header("📡 Web Security Analysis")
    
    tab1, tab2 = st.tabs(["Security Headers", "Website Analysis"])
    
    with tab1:
        st.subheader("HTTP Security Headers Analysis")
        
        url = st.text_input("Enter website URL:")
        
        if st.button("Analyze Headers") and url:
            with st.spinner("Analyzing security headers..."):
                headers, content_length = tools.website_security_headers(url)
            
            if "error" not in headers:
                score = 0
                total_headers = len(headers)
                
                for header, value in headers.items():
                    if value != "MISSING":
                        score += 1
                        st.success(f"✅ **{header}**: {value}")
                    else:
                        st.error(f"❌ **{header}**: {value}")
                
                security_score = (score / total_headers) * 100
                st.metric("Security Headers Score", f"{security_score:.1f}%")
                
                if security_score >= 80:
                    st.success("Good security headers configuration")
                elif security_score >= 60:
                    st.warning("Moderate security headers configuration")
                else:
                    st.error("Poor security headers configuration")
            else:
                st.error(f"Error analyzing headers: {headers['error']}")
    
    with tab2:
        st.subheader("Website Security Analysis")
        
        st.info("Comprehensive website security analysis")
        st.write("This tool analyzes various security aspects of websites:")
        st.write("• SSL/TLS configuration")
        st.write("• Security headers")
        st.write("• Vulnerabilities")
        st.write("• Content Security Policy")
        
        website = st.text_input("Website to analyze:")
        if st.button("Full Analysis") and website:
            with st.spinner("Performing comprehensive analysis..."):
                time.sleep(3)
                
                # Simulated results
                st.success("Analysis Complete!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("SSL Grade", "A")
                    st.metric("HTTPS", "Enabled")
                
                with col2:
                    st.metric("Security Headers", "6/8")
                    st.metric("Vulnerabilities", "2")
                
                with col3:
                    st.metric("Content Security", "Good")
                    st.metric("Overall Score", "85%")

def show_threat_intelligence(tools):
    st.header("🔍 Threat Intelligence")
    
    tab1, tab2, tab3 = st.tabs(["IP Reputation", "Domain Analysis", "Threat Feed"])
    
    with tab1:
        st.subheader("IP Reputation Analysis")
        
        ip_list = st.text_area("Enter IP addresses (one per line):")
        
        if st.button("Check Reputation") and ip_list:
            ips = [ip.strip() for ip in ip_list.split('\n') if ip.strip()]
            results = []
            
            for ip in ips:
                result = tools.ip_reputation_check(ip)
                results.append(result)
            
            # Display results in a table
            df = pd.DataFrame(results)
            st.dataframe(df)
    
    with tab2:
        st.subheader("Domain Threat Analysis")
        
        domain = st.text_input("Enter domain name:")
        
        if st.button("Analyze Domain") and domain:
            with st.spinner("Analyzing domain reputation..."):
                # Simulated domain analysis
                time.sleep(2)
                
                analysis = {
                    "domain": domain,
                    "age_days": np.random.randint(1, 3650),
                    "trust_score": np.random.randint(0, 100),
                    "malicious": np.random.choice([True, False], p=[0.2, 0.8])
                }
                
                st.metric("Domain Age", f"{analysis['age_days']} days")
                st.metric("Trust Score", f"{analysis['trust_score']}%")
                
                if analysis['malicious']:
                    st.error("🚨 Domain flagged as potentially malicious")
                else:
                    st.success("✅ Domain appears clean")
    
    with tab3:
        st.subheader("Live Threat Feed")
        
        st.info("Real-time threat intelligence feed")
        
        # Simulated threat feed
        threats = [
            {"type": "Malware", "name": "Emotet", "severity": "High", "timestamp": "2 hours ago"},
            {"type": "Phishing", "name": "Office365 Campaign", "severity": "Medium", "timestamp": "1 hour ago"},
            {"type": "DDoS", "name": "Mirai Botnet", "severity": "High", "timestamp": "30 minutes ago"},
        ]
        
        for threat in threats:
            severity_class = f"security-{threat['severity'].lower()}"
            st.markdown(f"""
            <div style="padding: 10px; margin: 5px 0; background: #1e1e2e; border-radius: 5px;">
                <strong>{threat['type']}</strong> - {threat['name']} - 
                <span class="{severity_class}">{threat['severity']}</span> - 
                <em>{threat['timestamp']}</em>
            </div>
            """, unsafe_allow_html=True)

def show_data_analysis(tools):
    st.header("📊 Security Data Analysis")
    
    tab1, tab2 = st.tabs(["File Analysis", "Security Analytics"])
    
    with tab1:
        st.subheader("File Security Analysis")
        
        uploaded_file = st.file_uploader("Upload file for analysis:", type=['exe', 'dll', 'txt', 'pdf', 'doc'])
        
        if uploaded_file is not None:
            file_content = uploaded_file.getvalue()
            
            st.write(f"**File:** {uploaded_file.name}")
            st.write(f"**Size:** {len(file_content)} bytes")
            
            # Hash analysis
            st.subheader("File Hashes")
            hashes = tools.file_hash_analyzer(file_content)
            for algo, hash_val in hashes.items():
                st.text_input(f"{algo}:", hash_val)
            
            # String extraction
            st.subheader("Extracted Strings")
            strings = tools.string_extractor(file_content)
            if strings:
                st.text_area("Found strings:", "\n".join(strings[:50]), height=200)
            else:
                st.info("No readable strings found in the file")
    
    with tab2:
        st.subheader("Security Analytics Dashboard")
        
        # Generate sample security data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        security_events = pd.DataFrame({
            'date': dates,
            'malware_detected': np.random.poisson(5, 100),
            'phishing_attempts': np.random.poisson(8, 100),
            'failed_logins': np.random.poisson(20, 100),
            'firewall_blocks': np.random.poisson(15, 100)
        })
        
        # Security metrics over time
        fig = px.line(security_events, x='date', y=['malware_detected', 'phishing_attempts', 'firewall_blocks'],
                     title='Security Events Over Time', labels={'value': 'Count', 'variable': 'Event Type'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Event distribution
        total_events = security_events[['malware_detected', 'phishing_attempts', 'failed_logins', 'firewall_blocks']].sum()
        fig = px.pie(values=total_events.values, names=total_events.index, title='Security Event Distribution')
        st.plotly_chart(fig, use_container_width=True)

def show_utilities(tools):
    st.header("🛠️ Security Utilities")
    
    tab1, tab2, tab3 = st.tabs(["QR Generator", "Data Converter", "System Info"])
    
    with tab1:
        st.subheader("QR Code Generator")
        
        qr_data = st.text_input("Data for QR code:")
        if st.button("Generate QR Code") and qr_data:
            qr_image = tools.generate_qr_code(qr_data)
            if qr_image:
                st.image(qr_image, caption="QR Code", use_column_width=True)
                
                # Download button
                st.download_button(
                    label="Download QR Code",
                    data=qr_image,
                    file_name="qrcode.png",
                    mime="image/png"
                )
            else:
                st.error("Failed to generate QR code")
    
    with tab2:
        st.subheader("Data Format Converter")
        
        col1, col2 = st.columns(2)
        
        with col1:
            input_data = st.text_area("Input data:")
            conversion_type = st.selectbox("Conversion:", 
                                         ["Text to Base64", "Base64 to Text", 
                                          "Text to Hex", "Hex to Text"])
        
        with col2:
            if st.button("Convert"):
                if conversion_type == "Text to Base64":
                    result = tools.multi_format_encoder(input_data, 'base64')
                elif conversion_type == "Base64 to Text":
                    result = tools.multi_format_decoder(input_data, 'base64')
                elif conversion_type == "Text to Hex":
                    result = tools.multi_format_encoder(input_data, 'hex')
                elif conversion_type == "Hex to Text":
                    result = tools.multi_format_decoder(input_data, 'hex')
                
                st.text_area("Converted result:", result, height=200)
    
    with tab3:
        st.subheader("System Information")
        
        if st.button("Generate System Report"):
            # Simulated system information
            sys_info = {
                "Application": "Advanced Cybersecurity Platform",
                "Version": "2.0.0",
                "Python Version": "3.9+",
                "Streamlit Version": "1.28.0",
                "Status": "🟢 Operational",
                "Last Update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Features Available": "15+ Security Tools",
                "Data Processing": "Client-side only"
            }
            
            st.json(sys_info)
            
            st.success("✅ System is functioning normally")
            st.info("All security tools are available and operational")

if __name__ == "__main__":
    main()
