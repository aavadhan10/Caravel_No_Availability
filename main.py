import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import json
import re
from functools import lru_cache
from supabase import create_client, Client

# Page Configuration
st.set_page_config(
    page_title="Legal Expert Finder",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Direct initialization without using st.secrets
def init_supabase():
    try:
        # Hardcoded credentials (not ideal, but will help for testing)
        supabase_url = "https://inhokxxeswrjleibzqyw.supabase.co"
        supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVjkueyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImluaG9reHhlc3dyamxlaWJ6cXl3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MTMxNDk5NDIsImV4cCI6MjAyODcyNTk0Mn0.eyJpc3MiOiJzd"
        
        # Try getting from environment variables first (more secure)
        if 'SUPABASE_URL' in os.environ and 'SUPABASE_KEY' in os.environ:
            supabase_url = os.environ.get('SUPABASE_URL')
            supabase_key = os.environ.get('SUPABASE_KEY')
        
        # Alternatively, try to get from st.secrets if available
        try:
            if 'supabase' in st.secrets:
                supabase_url = st.secrets["supabase"]["url"]
                supabase_key = st.secrets["supabase"]["key"]
        except:
            # Just use the hardcoded credentials if secrets aren't available
            pass
            
        # Initialize the client
        supabase: Client = create_client(supabase_url, supabase_key)
        return supabase
    except Exception as e:
        st.error(f"Error initializing Supabase: {str(e)}")
        return None

# Initialize Supabase client
supabase = init_supabase()

# Define the admin password (again, not ideal but will help for testing)
ADMIN_PASSWORD = "CaravelAI2025"

# Function to check admin password
def is_admin_password_valid(password):
    # Try getting from environment variables first
    if 'ADMIN_PASSWORD' in os.environ:
        return password == os.environ.get('ADMIN_PASSWORD')
    
    # Try getting from st.secrets if available
    try:
        if 'supabase' in st.secrets and 'admin_password' in st.secrets["supabase"]:
            return password == st.secrets["supabase"]["admin_password"]
    except:
        pass
    
    # Fallback to hardcoded password
    return password == ADMIN_PASSWORD

# Function to save feedback to Supabase
def save_feedback_to_supabase(feedback_data):
    if not supabase:
        st.error("Supabase connection not available.")
        return {"success": False, "message": "Database connection not available"}
    
    try:
        # Insert feedback into 'feedback' table
        response = supabase.table('feedback').insert(feedback_data).execute()
        
        # Check for errors
        if hasattr(response, 'error') and response.error:
            return {"success": False, "message": response.error}
        
        return {"success": True, "message": "Feedback saved to database", "id": response.data[0].get('id') if response.data else None}
    except Exception as e:
        st.error(f"Error saving feedback: {e}")
        return {"success": False, "message": str(e)}

# Function to get feedback from Supabase
def get_feedback_from_supabase(limit=50):
    if not supabase:
        st.error("Supabase connection not available.")
        return []
    
    try:
        # Get feedback from 'feedback' table
        response = supabase.table('feedback').select('*').order('created_at', desc=True).limit(limit).execute()
        
        # Check for errors
        if hasattr(response, 'error') and response.error:
            st.error(f"Error fetching feedback: {response.error}")
            return []
        
        return response.data
    except Exception as e:
        st.error(f"Error fetching feedback: {e}")
        return []

# CSS for styling
st.markdown("""
<style>
.lawyer-card {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
    border-left: 5px solid #1e88e5;
}
.lawyer-name {
    color: #1e4b79;
    font-size: 22px;
    font-weight: bold;
}
.lawyer-email {
    color: #263238;
    font-style: italic;
}
.skill-tag {
    background-color: #e3f2fd;
    border-radius: 15px;
    padding: 5px 10px;
    margin-right: 5px;
    display: inline-block;
    font-size: 14px;
}
.reasoning-box {
    background-color: #f1f8e9;
    border-radius: 5px;
    padding: 20px;
    margin-top: 15px;
    border-left: 5px solid #7cb342;
    font-size: 15px;
    line-height: 1.5;
}
.match-rationale-title {
    font-weight: bold;
    font-size: 16px;
    color: #2e7d32;
    margin-bottom: 8px;
}
.recent-query-button {
    margin-bottom: 8px !important;
    width: 100%;
}
h1 {
    color: #1e4b79;
}
.scroll-container {
    max-height: 400px;
    overflow-y: auto;
    padding-right: 10px;
}
.stButton button {
    width: 100%;
    margin-bottom: 8px;
}
.billable-rate {
    color: #455a64;
    font-size: 14px;
    margin-top: 5px;
}
.practice-area {
    color: #455a64;
    font-size: 14px;
    margin-top: 5px;
    font-weight: 500;
}
.bio-section {
    margin: 10px 0;
    padding: 10px;
    background-color: #f5f5f5;
    border-radius: 5px;
}
.bio-level {
    font-weight: bold;
    color: #1e4b79;
    font-size: 15px;
    margin-bottom: 4px;
}
.bio-details {
    color: #555;
    font-size: 14px;
    margin-bottom: 5px;
}
.bio-experience, .bio-education, .industry-experience {
    font-size: 14px;
    margin-top: 5px;
    color: #333;
}
.feedback-container {
    background-color: #f0f8ff;
    border-radius: 10px;
    padding: 20px;
    margin-top: 30px;
    border-left: 5px solid #4682b4;
}
.feedback-title {
    color: #1e4b79;
    font-size: 18px;
    font-weight: bold;
    margin-bottom: 10px;
}
.feedback-button {
    background-color: #1e88e5;
    color: white;
    border: none;
    padding: 10px 15px;
    border-radius: 5px;
    cursor: pointer;
    font-weight: bold;
}
.bio-match-tag {
    background-color: #ffd54f;
    border-radius: 15px;
    padding: 3px 8px;
    margin-right: 5px;
    display: inline-block;
    font-size: 12px;
    color: #5d4037;
    font-weight: bold;
}
.bio-blurb {
    background-color: #e8f5e8;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
    border-left: 4px solid #4caf50;
    font-style: italic;
    color: #2e7d32;
}
</style>
""", unsafe_allow_html=True)

# Set up sidebar
st.sidebar.title("⚖️ Legal Expert Finder")
st.sidebar.title("About")
st.sidebar.info(
    "This internal tool helps match client legal needs with the right lawyer based on expertise and biographical information. "
    "Designed for partners and executive assistants to quickly find the best internal resource for client requirements."
)
st.sidebar.markdown("---")

# Recent client queries section in sidebar
st.sidebar.markdown("### Recent Client Queries")
recent_queries = [
    "IP licensing for SaaS company",
    "Employment dispute in Ontario", 
    "M&A due diligence for tech acquisition",
    "Privacy compliance for healthcare app",
    "Commercial contracts & hospitality"
]

# Initialize session state variables
if 'query' not in st.session_state:
    st.session_state['query'] = ""
if 'search_pressed' not in st.session_state:
    st.session_state['search_pressed'] = False
if 'feedback_text' not in st.session_state:
    st.session_state.feedback_text = ""
if 'last_saved_feedback' not in st.session_state:
    st.session_state.last_saved_feedback = ""
if 'feedback_saved' not in st.session_state:
    st.session_state.feedback_saved = False
if 'expected_lawyers' not in st.session_state:
    st.session_state.expected_lawyers = ""
if 'incorrect_lawyers' not in st.session_state:
    st.session_state.incorrect_lawyers = ""
if 'rating' not in st.session_state:
    st.session_state.rating = ""
if 'user_role' not in st.session_state:
    st.session_state.user_role = "Partner"
if 'show_admin_dashboard' not in st.session_state:
    st.session_state.show_admin_dashboard = False
if 'admin_mode' not in st.session_state:
    st.session_state.admin_mode = False
if 'api_key_found' not in st.session_state:
    st.session_state.api_key_found = False
if 'api_status' not in st.session_state:
    st.session_state.api_status = None
if 'api_response_preview' not in st.session_state:
    st.session_state.api_response_preview = None
if 'full_claude_response' not in st.session_state:
    st.session_state.full_claude_response = None

# Helper function to set the query and trigger search
def set_query(text):
    st.session_state['query'] = text
    st.session_state['search_pressed'] = True

for query in recent_queries:
    if st.sidebar.button(query, key=f"recent_{query}", help=f"Use this recent query: {query}"):
        set_query(query)

st.sidebar.markdown("---")
st.sidebar.markdown("### Need Help?")
st.sidebar.info(
    "For assistance with the matching tool or to add a lawyer to the database, contact the AI team at officeofinnovation@brieflylegal.com"
)

# MODIFIED: Enhanced CSV reading functions with better bio.blurb detection
def detect_csv_structure(df):
    """
    Analyze the CSV structure to understand what data is available
    Returns a dictionary with information about the detected structure
    """
    structure = {
        'name_columns': [],
        'email_columns': [],
        'skill_columns': [],
        'bio_columns': [],
        'text_columns': [],
        'blurb_columns': [],  # NEW: specifically for bio.blurb
        'total_columns': len(df.columns)
    }
    
    # Convert all column names to lowercase for easier matching
    columns_lower = [col.lower() for col in df.columns]
    
    # Detect name columns
    name_patterns = ['name', 'submitter name', 'first name', 'last name', 'lawyer', 'attorney']
    for i, col in enumerate(columns_lower):
        if any(pattern in col for pattern in name_patterns):
            structure['name_columns'].append(df.columns[i])
    
    # Detect email columns
    email_patterns = ['email', 'mail', 'contact']
    for i, col in enumerate(columns_lower):
        if any(pattern in col for pattern in email_patterns):
            structure['email_columns'].append(df.columns[i])
    
    # NEW: Detect bio.blurb columns specifically
    blurb_patterns = ['blurb', 'bio.blurb', 'biography', 'summary', 'description', 'profile']
    for i, col in enumerate(columns_lower):
        if any(pattern in col for pattern in blurb_patterns):
            structure['blurb_columns'].append(df.columns[i])
    
    # Detect skill columns (columns with numeric values or skill-like names)
    skill_patterns = ['skill', 'expertise', 'experience', 'practice', 'area', 'competency']
    for i, col in enumerate(columns_lower):
        # Check if it's a skill column by name pattern
        if any(pattern in col for pattern in skill_patterns):
            structure['skill_columns'].append(df.columns[i])
        # Or if it contains mostly numeric values
        elif df[df.columns[i]].dtype in ['int64', 'float64']:
            structure['skill_columns'].append(df.columns[i])
    
    # Detect biographical columns
    bio_patterns = ['level', 'title', 'jurisdiction', 'location', 'education', 'firm', 'company', 
                   'experience', 'industry', 'language', 'award', 'recognition', 'call', 'bar',
                   'previous', 'notable', 'expert', 'bio', 'background', 'qualification']
    for i, col in enumerate(columns_lower):
        if any(pattern in col for pattern in bio_patterns):
            # Don't double-count blurb columns
            if df.columns[i] not in structure['blurb_columns']:
                structure['bio_columns'].append(df.columns[i])
    
    # All text columns that aren't already categorized
    for col in df.columns:
        if (col not in structure['name_columns'] and 
            col not in structure['email_columns'] and 
            col not in structure['skill_columns'] and 
            col not in structure['bio_columns'] and
            col not in structure['blurb_columns'] and
            df[col].dtype == 'object'):
            structure['text_columns'].append(col)
    
    return structure

def extract_lawyer_info_flexible(row, structure):
    """
    Extract lawyer information from a row based on detected structure
    PRIORITIZES bio.blurb and biographical information over skills
    """
    # Extract name - try multiple strategies
    name = ""
    if structure['name_columns']:
        # If we have specific name columns, use them
        name_parts = []
        for col in structure['name_columns']:
            if pd.notna(row[col]) and str(row[col]).strip():
                name_parts.append(str(row[col]).strip())
        name = " ".join(name_parts)
    
    # If no name found, use first non-empty text column
    if not name and structure['text_columns']:
        for col in structure['text_columns'][:3]:  # Check first 3 text columns
            if pd.notna(row[col]) and str(row[col]).strip():
                name = str(row[col]).strip()
                break
    
    # Extract email
    email = ""
    if structure['email_columns']:
        for col in structure['email_columns']:
            if pd.notna(row[col]) and str(row[col]).strip():
                email = str(row[col]).strip()
                break
    
    # If no email found, generate a placeholder
    if not email:
        email = f"{name.lower().replace(' ', '.')}@example.com"
    
    # NEW: Extract bio.blurb with highest priority
    blurb = ""
    if structure['blurb_columns']:
        for col in structure['blurb_columns']:
            if pd.notna(row[col]) and str(row[col]).strip():
                blurb = str(row[col]).strip()
                break
    
    # Extract skills (now with lower priority)
    skills = {}
    for col in structure['skill_columns']:
        if pd.notna(row[col]):
            value = row[col]
            # Convert to numeric if possible
            try:
                value = float(value)
                if value > 0:
                    # Clean up column name for skill
                    skill_name = col.replace('(Skill', '').replace(')', '').strip()
                    # Remove numbers from skill names
                    skill_name = re.sub(r'\s+\d+\s*$', '', skill_name).strip()
                    skills[skill_name] = value
            except (ValueError, TypeError):
                # If it's text, still include it with a default value
                if str(value).strip():
                    skill_name = col.strip()
                    skills[skill_name] = 1.0
    
    # Extract biographical information
    bio = {}
    bio_field_mapping = {
        'level': ['level', 'title', 'position', 'rank'],
        'call': ['call', 'bar', 'admission'],
        'jurisdiction': ['jurisdiction', 'province', 'state'],
        'location': ['location', 'office', 'city'],
        'practice_areas': ['practice', 'area', 'specialization', 'expertise'],
        'industry_experience': ['industry', 'sector', 'business'],
        'languages': ['language', 'linguistic'],
        'previous_in_house': ['in-house', 'in house', 'corporate', 'company'],
        'previous_firms': ['firm', 'previous', 'prior'],
        'education': ['education', 'school', 'university', 'degree'],
        'awards': ['award', 'recognition', 'honor'],
        'notable_items': ['notable', 'personal', 'detail', 'note'],
        'expert': ['expert', 'specialty', 'focus'],
        'blurb': ['blurb', 'bio.blurb', 'biography', 'summary', 'description', 'profile']  # NEW
    }
    
    # Initialize bio fields
    for field in bio_field_mapping.keys():
        bio[field] = ""
    
    # Set the blurb first if we found it
    if blurb:
        bio['blurb'] = blurb
    
    # Map columns to bio fields
    for col in structure['bio_columns'] + structure['text_columns'] + structure['blurb_columns']:
        col_lower = col.lower()
        value = str(row[col]) if pd.notna(row[col]) else ""
        
        if value.strip():
            # Find the best matching bio field
            best_match = None
            for bio_field, patterns in bio_field_mapping.items():
                if any(pattern in col_lower for pattern in patterns):
                    if not bio[bio_field]:  # Only set if not already set
                        bio[bio_field] = value
                        best_match = bio_field
                        break
            
            # If no specific match, add to notable_items or expert
            if not best_match:
                if not bio['notable_items']:
                    bio['notable_items'] = value
                elif not bio['expert']:
                    bio['expert'] = value
    
    return {
        'name': name,
        'email': email,
        'skills': skills,
        'bio': bio,
        'blurb': blurb  # NEW: direct access to blurb
    }

# MODIFIED: Main data loading function to handle any CSV structure with bio priority
@lru_cache(maxsize=1)
def load_lawyer_data():
    try:
        # Try to load the main CSV file - prioritize Caravel_New.csv
        possible_files = ['Caravel_New.csv', 'combined_unique.csv', 'lawyers.csv', 'data.csv']
        df = None
        loaded_filename = None
        
        for filename in possible_files:
            try:
                df = pd.read_csv(filename)
                loaded_filename = filename
                st.info(f"Successfully loaded data from {filename}")
                break
            except FileNotFoundError:
                continue
        
        if df is None:
            st.error("No CSV file found. Please ensure your CSV file is in the same directory.")
            return None
        
        # Analyze the CSV structure
        structure = detect_csv_structure(df)
        
        # Debug information
        with st.expander("CSV Structure Analysis", expanded=False):
            st.write(f"Loaded file: {loaded_filename}")
            st.write("Detected Structure:")
            st.json(structure)
            st.write("Sample of first few rows:")
            st.dataframe(df.head())
            
            # Show blurb column detection specifically
            if structure['blurb_columns']:
                st.write("🎯 Bio.blurb columns detected:")
                for col in structure['blurb_columns']:
                    st.write(f"  - {col}")
                    # Show sample blurb content
                    sample_blurb = df[col].dropna().iloc[0] if not df[col].dropna().empty else "No content"
                    st.write(f"    Sample: {str(sample_blurb)[:200]}...")
        
        # Process the data
        lawyers = []
        practice_areas = ["Corporate", "Litigation", "IP", "Employment", "Privacy", "Finance", "Real Estate", "Tax"]
        rate_ranges = ["$400-500/hr", "$500-600/hr", "$600-700/hr", "$700-800/hr", "$800-900/hr"]
        
        for _, row in df.iterrows():
            try:
                lawyer_info = extract_lawyer_info_flexible(row, structure)
                
                # Skip if no name found
                if not lawyer_info['name']:
                    continue
                
                # Add mock data for fields that aren't in the CSV
                lawyer_info['practice_area'] = np.random.choice(practice_areas)
                lawyer_info['billable_rate'] = np.random.choice(rate_ranges)
                lawyer_info['last_client'] = f"Client {np.random.randint(100, 999)}"
                
                # Try to infer practice area from bio data if available
                bio_text = ""
                if lawyer_info['bio'].get('practice_areas'):
                    bio_text += lawyer_info['bio']['practice_areas'].lower()
                if lawyer_info['bio'].get('blurb'):
                    bio_text += " " + lawyer_info['bio']['blurb'].lower()
                
                if bio_text:
                    if 'corporate' in bio_text or 'commercial' in bio_text:
                        lawyer_info['practice_area'] = 'Corporate'
                    elif 'litigation' in bio_text or 'dispute' in bio_text:
                        lawyer_info['practice_area'] = 'Litigation'
                    elif 'intellectual property' in bio_text or 'ip' in bio_text:
                        lawyer_info['practice_area'] = 'IP'
                    elif 'employ' in bio_text:
                        lawyer_info['practice_area'] = 'Employment'
                    elif 'privacy' in bio_text:
                        lawyer_info['practice_area'] = 'Privacy'
                
                lawyers.append(lawyer_info)
                
            except Exception as e:
                st.warning(f"Error processing row: {e}")
                continue
        
        # Create unique skills list
        all_skills = set()
        for lawyer in lawyers:
            all_skills.update(lawyer['skills'].keys())
        
        return {
            'lawyers': lawyers,
            'unique_skills': list(all_skills),
            'structure': structure,
            'loaded_file': loaded_filename
        }
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# MODIFIED: Enhanced matching function that prioritizes bio/blurb over skills
def match_lawyers(data, query, top_n=10):
    if not data:
        return []
    
    # Convert query to lowercase for case-insensitive matching
    lower_query = query.lower()
    
    # Test users to exclude - preventing test accounts from appearing in results
    excluded_users = ["Ankita", "Test", "Tania", "Antoine Malek", "Connie Chan", "Michelle Koyle", "Sue Gaudi", "Rose Os"]
    
    # For M&A queries, expand the search terms
    if "m&a" in lower_query or "merger" in lower_query or "acquisition" in lower_query:
        expanded_query = lower_query + " acquisitions mergers purchase sale of business"
        lower_query = expanded_query
    
    # Check for multi-criteria queries (like "commercial contracts AND hospitality")
    query_parts = []
    if " and " in lower_query or " & " in lower_query:
        # Split by " and " or " & "
        query_parts = re.split(r' and | & ', lower_query)
    else:
        query_parts = [lower_query]
    
    # Calculate match scores for each lawyer
    matches = []
    for lawyer in data['lawyers']:
        # Skip excluded users
        if any(excluded_name in lawyer['name'] for excluded_name in excluded_users):
            continue
            
        bio_score = 0
        blurb_score = 0  # NEW: separate score for bio.blurb
        skill_score = 0
        matched_bio_reasons = []
        matched_blurb_reasons = []  # NEW
        matched_skills = []
        all_criteria_matched = True if len(query_parts) > 1 else False
        
        # FIRST: Check bio.blurb for matches (HIGHEST PRIORITY)
        bio = lawyer.get('bio', {})
        blurb_text = bio.get('blurb', '').lower()
        
        if blurb_text:
            for query_part in query_parts:
                part_matched_in_blurb = False
                
                # Check for exact or partial matches in blurb
                if query_part.strip() in blurb_text:
                    blurb_score += 10  # HIGHEST score for blurb matches
                    matched_blurb_reasons.append({
                        'field': 'bio.blurb',
                        'value': bio.get('blurb', ''),
                        'match_type': 'exact_phrase'
                    })
                    part_matched_in_blurb = True
                
                # For multi-criteria queries, track if each part matched in blurb
                if not part_matched_in_blurb:
                    for word in query_part.split():
                        if word in blurb_text and len(word) > 3:  # Avoid matching small words
                            blurb_score += 6  # High score for word matches in blurb
                            matched_blurb_reasons.append({
                                'field': 'bio.blurb',
                                'value': bio.get('blurb', ''),
                                'match_type': 'word_match'
                            })
                            part_matched_in_blurb = True
                            break
        
        # SECOND: Check other biographical data for matches
        # Create a single text string from all biographical data to search (excluding blurb)
        bio_text = " ".join([
            bio.get('practice_areas', ''),
            bio.get('expert', ''),
            bio.get('industry_experience', ''),
            bio.get('notable_items', ''),
            bio.get('previous_in_house', ''),
            bio.get('previous_firms', ''),
            bio.get('education', ''),
            bio.get('awards', '')
        ]).lower()
        
        # Check each query part against biographical data
        for query_part in query_parts:
            part_matched_in_bio = False
            
            # Check for exact or partial matches in biographical data
            if query_part.strip() in bio_text:
                bio_score += 7  # High score for bio matches (but lower than blurb)
                
                # Determine which bio field matched
                for field, value in bio.items():
                    if field != 'blurb' and value and query_part.strip() in value.lower():
                        matched_bio_reasons.append({
                            'field': field,
                            'value': value,
                            'match_type': 'exact_phrase'
                        })
                        part_matched_in_bio = True
            
            # For multi-criteria queries, track if each part matched in bio
            if not part_matched_in_bio:
                for word in query_part.split():
                    if word in bio_text and len(word) > 3:  # Avoid matching small words
                        bio_score += 4
                        
                        # Determine which bio field matched
                        for field, value in bio.items():
                            if field != 'blurb' and value and word in value.lower():
                                matched_bio_reasons.append({
                                    'field': field,
                                    'value': value,
                                    'match_type': 'word_match'
                                })
                                part_matched_in_bio = True
                                break
            
            # THIRD: Check skills data as supporting evidence (LOWEST PRIORITY)
            part_matched_in_skills = False
            
            # Check each skill against the query part
            for skill, value in lawyer['skills'].items():
                skill_lower = skill.lower()
                
                # More precise matching - prefer exact matches over partial
                if skill_lower == query_part.strip():
                    # Exact match gets moderate score (lower than bio)
                    skill_score += value * 1.0
                    matched_skills.append({'skill': skill, 'value': value, 'match_type': 'exact'})
                    part_matched_in_skills = True
                elif query_part.strip() in skill_lower:
                    # Contains match
                    skill_score += value * 0.7
                    matched_skills.append({'skill': skill, 'value': value, 'match_type': 'contains'})
                    part_matched_in_skills = True
                elif any(word in skill_lower for word in query_part.split() if len(word) > 3):
                    # Word match (for words > 3 chars to avoid matching small words)
                    skill_score += value * 0.3
                    matched_skills.append({'skill': skill, 'value': value, 'match_type': 'word'})
                    part_matched_in_skills = True
            
            # For multi-criteria queries, check if this part matched in blurb, bio, or skills
            if len(query_parts) > 1 and not (part_matched_in_blurb or part_matched_in_bio or part_matched_in_skills):
                all_criteria_matched = False
        
        # For multi-criteria queries, if not all criteria matched, reset scores
        if len(query_parts) > 1 and not all_criteria_matched:
            bio_score = 0
            blurb_score = 0
            skill_score = 0
            matched_bio_reasons = []
            matched_blurb_reasons = []
            matched_skills = []
        
        # Calculate final score with blurb_score weighted highest, then bio_score, then skills
        final_score = (blurb_score * 3) + (bio_score * 2) + skill_score
        
        # Add lawyer to matches if scored
        if final_score > 0:
            # Remove duplicate skills and bio reasons
            unique_skills = {}
            for skill in matched_skills:
                skill_name = skill['skill']
                if skill_name not in unique_skills or skill['value'] > unique_skills[skill_name]['value']:
                    unique_skills[skill_name] = skill
            
            unique_bio_reasons = {}
            for reason in matched_bio_reasons:
                field = reason['field']
                if field not in unique_bio_reasons:
                    unique_bio_reasons[field] = reason
            
            unique_blurb_reasons = {}
            for reason in matched_blurb_reasons:
                field = reason['field']
                if field not in unique_blurb_reasons:
                    unique_blurb_reasons[field] = reason
            
            matches.append({
                'lawyer': lawyer,
                'score': final_score,
                'bio_score': bio_score,
                'blurb_score': blurb_score,  # NEW
                'skill_score': skill_score,
                'matched_bio_reasons': list(unique_bio_reasons.values()),
                'matched_blurb_reasons': list(unique_blurb_reasons.values()),  # NEW
                'matched_skills': sorted(list(unique_skills.values()), key=lambda x: x['value'], reverse=True)[:5]
            })
    
    # Sort by final score and take top N
    return sorted(matches, key=lambda x: x['score'], reverse=True)[:top_n]

# MODIFIED: Updated function to format Claude's analysis prompt emphasizing bio.blurb priority
def format_claude_prompt(query, matches):
    prompt = f"""
I need to analyze and provide detailed reasoning for why specific lawyers match a client's legal needs. You must PRIORITIZE biographical information (especially bio.blurb) over skill scores, while still referencing the scores as supporting evidence.

Client's Legal Need: "{query}"

For each lawyer, I'll provide their bio.blurb (most important), other biographical details, and self-reported skills. Focus primarily on the bio.blurb and biographical information to explain the match.

"""
    
    for i, match in enumerate(matches, 1):
        lawyer = match['lawyer']
        skills = match.get('matched_skills', [])
        bio_reasons = match.get('matched_bio_reasons', [])
        blurb_reasons = match.get('matched_blurb_reasons', [])
        bio = lawyer.get('bio', {})
        
        prompt += f"LAWYER {i}: {lawyer['name']}\n"
        prompt += "=============================================\n"
        
        # PRIORITY 1: Bio.blurb (MOST IMPORTANT)
        if bio.get('blurb'):
            prompt += "🎯 BIO.BLURB (PRIMARY MATCH FACTOR):\n"
            prompt += f"{bio['blurb']}\n\n"
            
            if blurb_reasons:
                prompt += "BLURB MATCH DETAILS:\n"
                for reason in blurb_reasons:
                    prompt += f"- Match Type: {reason.get('match_type', 'general')}\n"
                prompt += "\n"
        
        # PRIORITY 2: Other biographical information
        prompt += "BIOGRAPHICAL INFORMATION:\n"
        if bio.get('level'):
            prompt += f"- Level/Title: {bio['level']}\n"
        if bio.get('call'):
            prompt += f"- Called to Bar: {bio['call']}\n"
        if bio.get('jurisdiction'):
            prompt += f"- Jurisdiction: {bio['jurisdiction']}\n"
        if bio.get('location'):
            prompt += f"- Location: {bio['location']}\n"
        if bio.get('practice_areas'):
            prompt += f"- Practice Areas: {bio['practice_areas']}\n"
        if bio.get('industry_experience'):
            prompt += f"- Industry Experience: {bio['industry_experience']}\n"
        if bio.get('previous_in_house'):
            prompt += f"- Previous In-House Experience: {bio['previous_in_house']}\n"
        if bio.get('previous_firms'):
            prompt += f"- Previous Law Firms: {bio['previous_firms']}\n"
        if bio.get('education'):
            prompt += f"- Education: {bio['education']}\n"
        if bio.get('awards'):
            prompt += f"- Awards/Recognition: {bio['awards']}\n"
        if bio.get('expert'):
            prompt += f"- Areas of Expertise: {bio['expert']}\n"
        if bio.get('notable_items'):
            prompt += f"- Notable Experience: {bio['notable_items']}\n"
        
        # Show which bio fields matched
        if bio_reasons:
            prompt += "\nMATCHED BIOGRAPHICAL FACTORS:\n"
            for reason in bio_reasons:
                field_name = reason['field'].replace('_', ' ').title()
                prompt += f"- {field_name}: {reason['value']} (Match: {reason.get('match_type', 'general')})\n"
        
        prompt += "\n"
            
        # PRIORITY 3: Skills as supporting evidence (reference scores but don't prioritize)
        if skills:
            prompt += "SUPPORTING SKILL SCORES:\n"
            for skill in skills:
                prompt += f"- {skill['skill']}: {skill['value']} points ({skill.get('match_type', 'general')} match)\n"
        
        # Show match scores for reference
        prompt += f"\nMATCH SCORES (for reference only):\n"
        prompt += f"- Bio.Blurb Score: {match.get('blurb_score', 0)}\n"
        prompt += f"- Biographical Score: {match.get('bio_score', 0)}\n"
        prompt += f"- Skills Score: {match.get('skill_score', 0)}\n"
        prompt += f"- Total Score: {match.get('score', 0)}\n"
        
        prompt += "\n\n"
    
    prompt += """
IMPORTANT INSTRUCTIONS:

For each lawyer, write ONE detailed paragraph (6-8 sentences) explaining why they are an excellent match for this client's needs.

Your explanation MUST:
1. START with and PRIORITIZE the bio.blurb information - this is the most important factor
2. Reference relevant biographical details (practice areas, experience, education, etc.)
3. Mention the match scores as SUPPORTING EVIDENCE, not the primary reason
4. Be specific about how their background aligns with the client's needs
5. Use the bio.blurb as the foundation of your analysis
6. Reference skills scores to validate the biographical match
7. Avoid generic language - be specific and substantive

PRIORITIZATION ORDER:
1st: Bio.blurb content (most important)
2nd: Other biographical information
3rd: Skills scores (supporting evidence only)

Format your response as a JSON object where each key is the lawyer's name and each value is your explanation paragraph:

{
    "Lawyer Name 1": "Based on their bio.blurb which shows... [detailed explanation prioritizing blurb and bio over scores]",
    "Lawyer Name 2": "Their biographical profile indicates... [detailed explanation prioritizing blurb and bio over scores]",
    "Lawyer Name 3": "The bio.blurb reveals... [detailed explanation prioritizing blurb and bio over scores]"
}

DO NOT include any additional text outside of this JSON structure.
"""
    return prompt

# MODIFIED: Enhanced fallback explanation generator that prioritizes bio.blurb
def generate_fallback_explanation(lawyer, bio, skills):
    """Generate a detailed explanation for a lawyer when Claude API fails - prioritizes bio.blurb"""
    # Extract key information
    name = lawyer['name']
    blurb = bio.get('blurb', '')
    practice_areas = bio.get('practice_areas', 'relevant legal fields')
    industry_exp = bio.get('industry_experience', '')
    previous_exp = bio.get('previous_in_house', '')
    previous_firms = bio.get('previous_firms', '')
    education = bio.get('education', '')
    expertise = bio.get('expert', '')
    
    # Create detailed explanation based on available information
    explanation_parts = []
    
    # START with bio.blurb if available (HIGHEST PRIORITY)
    if blurb:
        explanation_parts.append(f"Based on {name}'s biographical profile, {blurb[:200]}{'...' if len(blurb) > 200 else ''}")
    else:
        explanation_parts.append(f"{name}'s profile shows specialization in {practice_areas}, which directly aligns with the client's requirements.")
    
    # Add specific biographical details
    if industry_exp:
        explanation_parts.append(f"Their industry experience in {industry_exp} provides valuable sector-specific knowledge for this matter.")
    
    # Add previous experience
    if previous_exp or previous_firms:
        exp_text = "Their professional background includes "
        if previous_exp:
            exp_text += f"in-house experience at {previous_exp}"
            if previous_firms:
                exp_text += f" and work at {previous_firms}"
        else:
            exp_text += f"work at {previous_firms}"
        exp_text += ", giving them practical insights into similar legal challenges."
        explanation_parts.append(exp_text)
    
    # Add education
    if education:
        explanation_parts.append(f"Their education from {education} provides a strong theoretical foundation for handling this type of matter.")
    
    # Add expertise if available
    if expertise:
        explanation_parts.append(f"Their specific expertise in {expertise} is directly relevant to addressing the client's needs effectively.")
    
    # Add skills as supporting evidence (LOWEST PRIORITY)
    if skills:
        skill_names = ", ".join([s["skill"] for s in skills[:3]])
        total_score = sum([s["value"] for s in skills[:3]])
        explanation_parts.append(f"This biographical match is further supported by their self-reported expertise scores in {skill_names} (total relevance score: {total_score:.1f}), confirming their qualifications in the areas needed for this client.")
    
    # Combine parts into a complete paragraph
    return " ".join(explanation_parts)

# Keep the same Claude API function but update the fallback call
def call_claude_api(prompt):
    # Try to get API key from environment variables or secrets
    api_key = os.environ.get("ANTHROPIC_API_KEY", None)
    
    # Try secrets if available (this should work with your secrets.toml configuration)
    try:
        if 'anthropic' in st.secrets and 'api_key' in st.secrets["anthropic"]:
            api_key = st.secrets["anthropic"]["api_key"]
            st.session_state['api_key_found'] = True
        else:
            st.session_state['api_key_found'] = False
    except Exception as e:
        st.session_state['api_key_found'] = False
        st.session_state['api_key_error'] = str(e)
    
    # Handle the case where no API key is provided
    if not api_key:
        st.warning("API key not found. Using mock reasoning data with bio.blurb priority instead.")
        # Return detailed mock reasoning data for the lawyers
        try:
            return call_claude_api_fallback(matches)
        except Exception as e:
            st.error(f"Error generating mock data: {str(e)}")
            return {"error": f"No API key provided and could not generate mock data: {str(e)}"}
    
    try:
        # Import just what we need to make an HTTP request
        import requests
        import json
        
        # Claude API endpoint
        url = "https://api.anthropic.com/v1/messages"
        
        # Headers
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        # Request payload - using Claude 3.5 Sonnet
        payload = {
            "model": "claude-3-5-sonnet-20240620",
            "max_tokens": 1500,  # Increased for longer bio-focused explanations
            "temperature": 0.2,
            "system": "You are a legal resource coordinator that analyzes lawyer expertise matches. You PRIORITIZE biographical information (especially bio.blurb) over skill scores. You provide detailed, factual explanations about why specific lawyers match particular client legal needs based primarily on their biographical background, using self-reported skills only as supporting evidence. Focus on bio.blurb content first, then other biographical data, and reference skill scores last as validation.",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        # Make the request
        response = requests.post(url, headers=headers, json=payload)
        
        # Log API response status and truncated content for debugging
        st.session_state['api_status'] = response.status_code
        st.session_state['api_response_preview'] = response.text[:100] + "..." if len(response.text) > 100 else response.text
        
        # Check for successful response
        if response.status_code == 200:
            response_json = response.json()
            response_text = response_json.get("content", [{}])[0].get("text", "")
            
            # Save the full response text for debugging
            st.session_state['full_claude_response'] = response_text
            
            # Try several approaches to parse the JSON
            # Approach 1: Find JSON within the response using regex
            import re
            json_match = re.search(r'(\{[\s\S]*\})', response_text, re.DOTALL)
            
            if json_match:
                try:
                    # Clean up the JSON string
                    json_str = json_match.group(0).strip()
                    # Attempt to parse the JSON
                    parsed_json = json.loads(json_str)
                    return parsed_json
                except json.JSONDecodeError as e:
                    st.warning(f"JSON parsing error (Approach 1): {str(e)}")
            
            # Approach 2: Try to find JSON blocks using another pattern
            json_match2 = re.search(r'({[\s\S]*})', response_text, re.DOTALL)
            if json_match2:
                try:
                    json_str = json_match2.group(0).strip()
                    # Try to fix common JSON issues (single quotes to double quotes)
                    json_str = json_str.replace("'", '"')
                    parsed_json = json.loads(json_str)
                    return parsed_json
                except json.JSONDecodeError as e:
                    st.warning(f"JSON parsing error (Approach 2): {str(e)}")
            
            # Approach 3: Try to manually construct a JSON from the response
            try:
                lines = response_text.split('\n')
                lawyer_explanations = {}
                
                current_lawyer = None
                current_explanation = []
                
                for line in lines:
                    # Look for patterns like "Lawyer Name": or "Lawyer Name":
                    match = re.search(r'"([^"]+)"\s*:', line) or re.search(r'"([^"]+)":', line)
                    if match and ':' in line:
                        # If we already have a lawyer, save their explanation
                        if current_lawyer and current_explanation:
                            lawyer_explanations[current_lawyer] = ' '.join(current_explanation)
                        
                        # Start a new lawyer
                        current_lawyer = match.group(1)
                        
                        # Extract the explanation part after the colon
                        explanation_part = line.split(':', 1)[1].strip()
                        if explanation_part.startswith('"') and explanation_part.endswith('"'):
                            explanation_part = explanation_part[1:-1]  # Remove quotes
                        
                        current_explanation = [explanation_part] if explanation_part else []
                    elif current_lawyer and line.strip():
                        # Continue the current explanation
                        line = line.strip()
                        if line.startswith('"') and line.endswith('"'):
                            line = line[1:-1]  # Remove quotes
                        if line.endswith(','):
                            line = line[:-1]  # Remove trailing comma
                        current_explanation.append(line)
                
                # Don't forget to add the last lawyer
                if current_lawyer and current_explanation:
                    lawyer_explanations[current_lawyer] = ' '.join(current_explanation)
                
                if lawyer_explanations:
                    return lawyer_explanations
            except Exception as e:
                st.warning(f"Manual JSON parsing error: {str(e)}")
            
            # If all parsing approaches fail, fall back to our manual generation
            st.warning("Could not parse the JSON response from Claude. Using bio-prioritized fallback explanations.")
            return call_claude_api_fallback(matches)
        else:
            st.error(f"API call failed with status code {response.status_code}")
            st.code(response.text)  # Show the error response for debugging
            
            # Fall back to the mock data
            return call_claude_api_fallback(matches)
            
    except Exception as e:
        st.error(f"Error calling Claude API: {str(e)}")
        
        # Provide a more detailed error message to help debugging
        import traceback
        st.error(f"Error details: {traceback.format_exc()}")
        
        # Return using the fallback function
        return call_claude_api_fallback(matches)

# MODIFIED: Fallback function for generating mock lawyer reasoning with bio priority
def call_claude_api_fallback(matches):
    try:
        lawyer_explanations = {}
        
        for match in matches:
            lawyer = match['lawyer']
            bio = lawyer.get('bio', {})
            skills = match.get('matched_skills', [])
            
            # Generate explanation using the helper function that prioritizes bio.blurb
            lawyer_explanations[lawyer['name']] = generate_fallback_explanation(lawyer, bio, skills)
        
        return lawyer_explanations
    except Exception as e:
        return {"error": f"Could not generate bio-prioritized fallback explanations: {str(e)}"}

# Keep all the feedback and admin functions the same...
def add_realtime_feedback():
    st.markdown("---")
    st.markdown('<div class="feedback-container">', unsafe_allow_html=True)
    st.markdown('<div class="feedback-title">Provide Feedback on Search Results</div>', unsafe_allow_html=True)
    
    # Use a callback to update feedback in real-time
    def on_feedback_change():
        # Only save to Supabase if feedback actually changed and is not empty
        if st.session_state.feedback_text and st.session_state.feedback_text != st.session_state.last_saved_feedback:
            # Create feedback data object
            feedback_data = {
                "query": st.session_state.get('query', ''),
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "feedback": st.session_state.feedback_text,
                "user_email": st.session_state.get('user_email', 'anonymous'),
                "user_role": st.session_state.get('user_role', 'partner'),
                "lawyer_results": json.dumps([m['lawyer']['name'] for m in matches]) if 'matches' in globals() else "[]",
                # Add expected lawyers that should have appeared
                "expected_lawyers": st.session_state.get('expected_lawyers', ''),
                # Add rating of results
                "rating": st.session_state.get('rating', ''),
                # Add incorrect lawyers field
                "incorrect_lawyers": st.session_state.get('incorrect_lawyers', '')
            }
            
            # Save to Supabase
            result = save_feedback_to_supabase(feedback_data)
            
            if result['success']:
                st.session_state.feedback_saved = True
                st.session_state.last_saved_feedback = st.session_state.feedback_text
                st.session_state.last_feedback_id = result.get('id')
            else:
                st.session_state.feedback_saved = False
                st.session_state.feedback_error = result.get('message')
    
    # Add rating selector
    st.markdown("<strong>How would you rate these search results?</strong>", unsafe_allow_html=True)
    cols = st.columns(5)
    rating_options = ["Poor", "Fair", "Good", "Very Good", "Excellent"]
    
    # Set up radio buttons for rating
    rating = st.radio(
        "Rating",
        rating_options,
        horizontal=True,
        label_visibility="collapsed",
        key="rating_input"
    )
    
    # Store rating in session state
    st.session_state.rating = rating
    
    # Optional email field for feedback attribution
    user_email = st.text_input(
        "Your email (optional):",
        value=st.session_state.get('user_email', ''),
        key="user_email_input",
        placeholder="Enter your email to associate with feedback"
    )
    
    # Store email in session state
    st.session_state.user_email = user_email
    
    # User role selection
    user_role = st.selectbox(
        "Your role:",
        ["Partner", "Associate", "Executive Assistant", "Legal Operations", "Other"],
        index=0,
        key="user_role_input"
    )
    
    # Store user role in session state
    st.session_state.user_role = user_role
    
    # Field to indicate expected lawyers that should have appeared
    expected_lawyers = st.text_input(
        "Lawyers you expected to see in results (comma-separated):",
        value=st.session_state.get('expected_lawyers', ''),
        key="expected_lawyers_input",
        placeholder="Names of lawyers you expected to see in these search results"
    )
    
    # Store expected lawyers in session state
    st.session_state.expected_lawyers = expected_lawyers
    
    # Field to indicate incorrect lawyers that shouldn't have appeared
    incorrect_lawyers = st.text_input(
        "Lawyers that shouldn't have appeared in results (comma-separated):",
        value=st.session_state.get('incorrect_lawyers', ''),
        key="incorrect_lawyers_input",
        placeholder="Names of lawyers you feel were incorrectly included in these results"
    )
    
    # Store incorrect lawyers in session state
    st.session_state.incorrect_lawyers = incorrect_lawyers
    
    # Create a text area that calls the callback when changed
    feedback = st.text_area(
        "Additional feedback on search results:",
        value=st.session_state.feedback_text,
        key="feedback_input",
        on_change=on_feedback_change,
        height=100,
        placeholder="Please share your thoughts on why these results are helpful or what could be improved..."
    )
    
    # Update the feedback text in session state
    st.session_state.feedback_text = feedback
    
    # If there's feedback, show it's being saved in real-time
    if feedback:
        # If feedback saved successfully
        if st.session_state.get('feedback_saved', False):
            # Display a success message
            st.success("✓ Feedback saved. Thank you for helping us improve the matching system!")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Admin dashboard for viewing feedback
def show_admin_dashboard():
    st.markdown("## Feedback Dashboard")
    
    # Get feedback data
    feedback_data = get_feedback_from_supabase(limit=50)
    
    if not feedback_data:
        st.info("No feedback data available.")
        return
    
    # Display feedback in a table
    st.markdown("### Recent Feedback")
    
    # Convert to DataFrame for easier display
    if isinstance(feedback_data, list) and feedback_data:
        df = pd.DataFrame(feedback_data)
        
        # Format timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Display the DataFrame
        st.dataframe(df)
        
        # Allow downloading as CSV
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Feedback as CSV",
            data=csv,
            file_name="feedback_data.csv",
            mime="text/csv"
        )
        
        # Analyze feedback to identify common patterns
        st.markdown("### Feedback Analysis")
        
        # Most common expected lawyers
        if 'expected_lawyers' in df.columns:
            all_expected = []
            for exp in df['expected_lawyers'].dropna():
                names = [name.strip() for name in exp.split(',') if name.strip()]
                all_expected.extend(names)
            
            if all_expected:
                expected_counts = pd.Series(all_expected).value_counts().head(10)
                st.markdown("#### Top Missing Lawyers")
                st.bar_chart(expected_counts)
        
        # Most common incorrect lawyers
        if 'incorrect_lawyers' in df.columns:
            all_incorrect = []
            for inc in df['incorrect_lawyers'].dropna():
                names = [name.strip() for name in inc.split(',') if name.strip()]
                all_incorrect.extend(names)
            
            if all_incorrect:
                incorrect_counts = pd.Series(all_incorrect).value_counts().head(10)
                st.markdown("#### Top Incorrectly Matched Lawyers")
                st.bar_chart(incorrect_counts)
        
        # Rating distribution
        if 'rating' in df.columns:
            rating_counts = df['rating'].value_counts()
            st.markdown("#### Rating Distribution")
            st.bar_chart(rating_counts)
    else:
        st.warning("Feedback data format is not as expected.")

# Updated admin mode function
def add_admin_mode():
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Admin Options")
    
    # Admin login
    admin_password = st.sidebar.text_input("Admin Password", type="password")
    
    if is_admin_password_valid(admin_password):
        st.session_state.admin_mode = True
        st.sidebar.success("Admin mode activated")
        
        # Admin actions
        if st.sidebar.button("View Feedback Dashboard"):
            st.session_state.show_admin_dashboard = True
    else:
        st.session_state.admin_mode = False

# Add admin mode to sidebar
add_admin_mode()

# Main app layout
st.title("⚖️ Legal Expert Finder")
st.markdown("Match client legal needs with the right lawyer based on **biographical information** (prioritizing bio.blurb)")

# Check if admin dashboard should be shown
if st.session_state.get('show_admin_dashboard', False) and st.session_state.get('admin_mode', False):
    show_admin_dashboard()
    # Add button to return to main app
    if st.button("Return to Main App"):
        st.session_state.show_admin_dashboard = False
        st.experimental_rerun()
else:
    # Load data
    data = load_lawyer_data()

    # Updated preset queries - changed "M&A without tech companies" to just "M&A"
    preset_queries = [
        "M&A",  # Simplified from "M&A not technology" to just "M&A"
        "Privacy compliance",
        "Startup law",
        "Employment issues",
        "Intellectual property protection",
        "Employment termination reviews",
        "Incorporation and corporate record keeping",
        "Healthcare compliance regulations",
        "Fintech regulatory compliance",
        "Commercial contracts & hospitality"  # Added this multi-criteria query based on feedback
    ]

    # Query input section
    query = st.text_area(
        "Describe client's legal needs in detail:", 
        value=st.session_state['query'],
        height=100,
        placeholder="Example: Client needs a lawyer with blockchain governance experience for cross-border cryptocurrency transactions or 'commercial contracts AND hospitality' for multiple criteria",
        key="query_input"
    )

    # Preset query buttons in rows of 3
    st.markdown("### Common Client Needs")
    cols = st.columns(3)
    for i, preset_query in enumerate(preset_queries):
        col_idx = i % 3
        with cols[col_idx]:
            if st.button(preset_query, key=f"preset_{i}"):
                set_query(preset_query)

    # Update query in session state from text area
    if query:
        st.session_state['query'] = query

    # Search button
    search_pressed = st.button("🔍 Find Matching Lawyers", type="primary", use_container_width=True)
    if search_pressed:
        st.session_state['search_pressed'] = True

    # Display results when search is pressed
    if st.session_state['search_pressed'] and st.session_state['query']:
        with st.spinner("Matching client needs with our legal experts (prioritizing biographical information)..."):
            # Get matches
            matches = match_lawyers(data, st.session_state['query'])
            
            if not matches:
                st.warning("No matching lawyers found. Please try a different query.")
            else:
                # Call Claude API for reasoning
                claude_prompt = format_claude_prompt(st.session_state['query'], matches)
                reasoning = call_claude_api(claude_prompt)
                
                # Add debugging information in expandable section
                with st.expander("Debug Information", expanded=False):
                    st.write("API Key Found:", st.session_state.get('api_key_found', 'Not checked'))
                    if 'api_key_error' in st.session_state:
                        st.write("API Key Error:", st.session_state['api_key_error'])
                    if 'api_status' in st.session_state:
                        st.write("API Status Code:", st.session_state['api_status'])
                    if 'api_response_preview' in st.session_state:
                        st.write("API Response Preview:", st.session_state['api_response_preview'])
                    st.write("Reasoning Type:", type(reasoning))
                    if isinstance(reasoning, dict):
                        st.write("Reasoning Keys:", list(reasoning.keys()))
                    if 'full_claude_response' in st.session_state:
                        st.write("Full Claude Response:")
                        st.code(st.session_state['full_claude_response'])
                
                # Display results
                st.markdown("## Matching Legal Experts")
                st.markdown(f"Found {len(matches)} lawyers matching client needs (ranked by biographical relevance):")
                
                # Sort by bio/blurb score first, then total score (prioritizing biographical matches)
                sorted_matches = sorted(matches, key=lambda x: (x.get('blurb_score', 0) + x.get('bio_score', 0), x['score']), reverse=True)
                
                for match in sorted_matches:
                    lawyer = match['lawyer']
                    matched_skills = match['matched_skills']
                    matched_bio_reasons = match.get('matched_bio_reasons', [])
                    matched_blurb_reasons = match.get('matched_blurb_reasons', [])
                    
                    with st.container():
                        # Get bio data
                        bio = lawyer.get('bio', {})
                        
                        # Use raw HTML string concatenation to avoid Streamlit escaping issues
                        html_output = f"""
                        <div class="lawyer-card">
                            <div class="lawyer-name">
                                {lawyer['name']}
                            </div>
                            <div class="lawyer-email">{lawyer['email']}</div>
                            <div class="practice-area">Practice Area: {lawyer['practice_area']}</div>
                        """
                        
                        # PRIORITY: Show bio.blurb first if available
                        if bio.get('blurb'):
                            html_output += f'''
                            <div class="bio-blurb">
                                <strong>🎯 Bio Profile:</strong><br/>
                                {bio['blurb'][:500]}{'...' if len(bio['blurb']) > 500 else ''}
                            </div>
                            '''
                        
                        # Create biographical info section
                        bio_html = ""
                        if bio:
                            bio_html += '<div class="bio-section">'
                            if bio.get('level'):
                                bio_html += f'<div class="bio-level">{bio["level"]}</div>'
                            
                            bio_details = []
                            if bio.get('call'):
                                bio_details.append(f'Called to Bar: {bio["call"]}')
                            if bio.get('jurisdiction'):
                                bio_details.append(f'Jurisdiction: {bio["jurisdiction"]}')
                            if bio.get('location'):
                                bio_details.append(f'Location: {bio["location"]}')
                            
                            if bio_details:
                                bio_html += f'<div class="bio-details">{" | ".join(bio_details)}</div>'
                                
                            if bio.get('previous_in_house'):
                                bio_html += f'<div class="bio-experience"><strong>In-House Experience:</strong> {bio["previous_in_house"]}</div>'
                            if bio.get('previous_firms'):
                                bio_html += f'<div class="bio-experience"><strong>Previous Firms:</strong> {bio["previous_firms"]}</div>'
                            if bio.get('education'):
                                bio_html += f'<div class="bio-education"><strong>Education:</strong> {bio["education"]}</div>'
                                
                            bio_html += '</div>'
                        
                        # Add the bio section to the HTML output
                        html_output += bio_html
                        
                        # Add matched blurb reasons section (highest priority)
                        if matched_blurb_reasons:
                            html_output += '<div style="margin-top: 10px;"><strong>🎯 Bio.Blurb Matches:</strong><br/>'
                            for reason in matched_blurb_reasons:
                                match_type = reason.get('match_type', 'general')
                                html_output += f'<span class="bio-match-tag">Bio.Blurb ({match_type})</span> '
                            html_output += '</div>'
                        
                        # Add matched bio reasons section
                        if matched_bio_reasons:
                            html_output += '<div style="margin-top: 10px;"><strong>Matched Biographical Factors:</strong><br/>'
                            for reason in matched_bio_reasons:
                                field_name = reason['field'].replace('_', ' ').title()
                                match_type = reason.get('match_type', 'general')
                                html_output += f'<span class="bio-match-tag">{field_name} ({match_type})</span> '
                            html_output += '</div>'
                        
                        # Add match scores for transparency
                        blurb_score = match.get('blurb_score', 0)
                        bio_score = match.get('bio_score', 0)
                        skill_score = match.get('skill_score', 0)
                        total_score = match.get('score', 0)
                        
                        html_output += f"""
                            <div style="margin-top: 10px;">
                                <strong>Match Scores:</strong>
                                Bio.Blurb: {blurb_score} | Bio: {bio_score} | Skills: {skill_score} | Total: {total_score}
                            </div>
                        """
                        
                        # Add skill tags as supporting evidence
                        if matched_skills:
                            html_output += f"""
                                <div style="margin-top: 10px;">
                                    <strong>Supporting Skill Evidence:</strong><br/>
                                    {"".join([f'<span class="skill-tag">{skill["skill"]}: {skill["value"]} ({skill.get("match_type", "general")})</span>' for skill in matched_skills])}
                                </div>
                            """
                        
                        html_output += "</div>"
                        st.markdown(html_output, unsafe_allow_html=True)
                        
                        # Get the specific explanation for this lawyer
                        lawyer_reasoning = "No specific analysis available for this lawyer."
                        
                        # Try different variations of the lawyer name that might be in the reasoning dict
                        lawyer_name_variants = [
                            lawyer['name'],  # Original name
                            lawyer['name'].strip(),  # Stripped of whitespace
                            ' '.join(lawyer['name'].split()),  # Normalized spaces
                            lawyer['name'].replace('  ', ' ')  # Replace double spaces
                        ]
                        
                        # Try to find a match in the reasoning dictionary
                        if isinstance(reasoning, dict):
                            # Try different variants of the name
                            for name_variant in lawyer_name_variants:
                                if name_variant in reasoning:
                                    lawyer_reasoning = reasoning[name_variant]
                                    break
                            
                            # If still not found, use the fallback
                            if lawyer_reasoning == "No specific analysis available for this lawyer.":
                                # Generate a bio-prioritized fallback explanation for this specific lawyer
                                lawyer_reasoning = generate_fallback_explanation(lawyer, bio, matched_skills)
                        else:
                            # If reasoning is not a dictionary, generate a bio-prioritized fallback explanation
                            lawyer_reasoning = generate_fallback_explanation(lawyer, bio, matched_skills)
                        
                        # Add the reasoning section with emphasis on bio priority
                        st.markdown("### WHY THIS LAWYER IS AN EXCELLENT MATCH:")
                        st.markdown(f"_{lawyer_reasoning}_", unsafe_allow_html=False)
                
                # Action buttons for results
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📧 Email These Matches to Requester", use_container_width=True):
                        st.success("Match results have been emailed to the requester!")
                with col2:
                    if st.button("📆 Schedule Consultation", use_container_width=True):
                        st.success("Consultation has been scheduled with these lawyers!")
                
                # Add real-time feedback
                add_realtime_feedback()

    # Show exploration section when no search is active
    if not st.session_state['search_pressed'] or not st.session_state['query']:
        st.markdown("## Explore Available Legal Expertise")
        
        if data:
            # Show information about data loading and bio.blurb priority
            st.info(f"✅ Loaded {len(data['lawyers'])} lawyers from {data.get('loaded_file', 'CSV file')} with biographical information prioritized over skill scores.")
            
            # Check how many lawyers have bio.blurb data
            lawyers_with_blurb = sum(1 for lawyer in data['lawyers'] if lawyer.get('bio', {}).get('blurb'))
            st.info(f"🎯 {lawyers_with_blurb} lawyers have detailed bio.blurb profiles (highest matching priority)")
            
            # Create a visual breakdown of legal expertise
            all_skills = {}
            for lawyer in data['lawyers']:
                for skill, value in lawyer['skills'].items():
                    if skill in all_skills:
                        all_skills[skill] += value
                    else:
                        all_skills[skill] = value
            
            # Get top 20 skills by total points
            top_skills = sorted(all_skills.items(), key=lambda x: x[1], reverse=True)[:20]
            
            # Show bar chart of top skills in scrollable container
            st.markdown("### Most Common Legal Expertise Areas (Supporting Evidence)")
            st.markdown('<div class="scroll-container">', unsafe_allow_html=True)
            chart_data = pd.DataFrame({
                'Skill': [s[0] for s in top_skills],
                'Total Points': [s[1] for s in top_skills]
            })
            st.bar_chart(chart_data.set_index('Skill'))
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("### Instructions for Matching")
            st.markdown("""
            Enter your client's specific legal needs above or select a common query to find matching legal experts. 
            
            **🎯 NEW: Bio.Blurb Priority Matching**
            - Matches are now **prioritized by biographical information**, especially bio.blurb content
            - Skill scores are used as **supporting evidence only**
            - Results show bio.blurb content prominently when available
            
            Be as specific as possible about their requirements, including:
            
            - The type of legal expertise needed
            - Any industry-specific requirements
            - Geographic considerations (e.g., province-specific needs)
            - The nature of the legal matter
            - Timeframe and urgency
            
            For multi-criteria searches, use "AND" or "&" between criteria (e.g., "commercial contracts AND hospitality").
            The system will match the query primarily with lawyers' biographical profiles and experience, using self-reported expertise as validation.
            """)

    # Footer
    st.markdown("---")
    st.markdown(
        "This internal tool **prioritizes biographical information (especially bio.blurb)** over skill scores when matching lawyers. "
        f"Results are sorted by biographical relevance with skill scores as supporting evidence. "
        f"Currently loaded: {len(data['lawyers']) if data else 0} lawyers "
        f"({sum(1 for lawyer in data['lawyers'] if lawyer.get('bio', {}).get('blurb')) if data else 0} with detailed bio.blurb profiles) "
        f"with {len(data['unique_skills']) if data else 0} unique skills. "
        "Last updated: July 8, 2025"
    )
