import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import json
import re
import html
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
    "For assistance with the matching tool or to add a lawyer to the database, contact the Legal Operations team at legalops@example.com"
)

# Function to load and process the CSV data
@lru_cache(maxsize=1)  # Cache the result to avoid reloading
def load_lawyer_data():
    try:
        # Load the skills data
        skills_df = pd.read_csv('combined_unique.csv')
        skills_data = process_lawyer_data(skills_df)
        
        # Load the biographical data
        bio_df = pd.read_csv('BD_Caravel.csv')
        bio_data = process_bio_data(bio_df)
        
        # Combine the data
        combined_data = combine_lawyer_data(skills_data, bio_data)
        
        # Verify and correct practice areas
        verified_data = verify_practice_areas(combined_data)
        
        return verified_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def process_lawyer_data(df):
    # Get all skill columns
    skill_columns = [col for col in df.columns if '(Skill' in col]
    
    # Create a map of normalized skill names
    skill_map = {}
    for col in skill_columns:
        match = re.match(r'(.*) \(Skill \d+\)', col)
        if match:
            skill_name = match.group(1)
            if skill_name not in skill_map:
                skill_map[skill_name] = []
            skill_map[skill_name].append(col)
    
    # Function to get max skill value across duplicate columns
    def get_max_skill_value(lawyer_row, skill_name):
        columns = skill_map.get(skill_name, [])
        values = [lawyer_row[col] for col in columns if pd.notna(lawyer_row[col])]
        return max(values) if values else 0
    
    # Create lawyer profiles with mock data for demo purposes
    lawyers = []
    practice_areas = ["Corporate", "Litigation", "IP", "Employment", "Privacy", "Finance", "Real Estate", "Tax"]
    rate_ranges = ["$400-500/hr", "$500-600/hr", "$600-700/hr", "$700-800/hr", "$800-900/hr"]
    
    for _, row in df.iterrows():
        lawyer_name = row['Submitter Name']
        
        profile = {
            'name': lawyer_name,
            'email': row['Submitter Email'],
            'skills': {},
            # Add mock data for other fields
            'practice_area': np.random.choice(practice_areas),
            'billable_rate': np.random.choice(rate_ranges),
            'last_client': f"Client {np.random.randint(100, 999)}"
        }
        
        # Extract skills with non-zero values
        for skill_name in skill_map:
            value = get_max_skill_value(row, skill_name)
            if value > 0:
                profile['skills'][skill_name] = value
        
        lawyers.append(profile)
    
    return {
        'lawyers': lawyers,
        'skill_map': skill_map,
        'unique_skills': list(skill_map.keys())
    }

# Function to process the biographical data
def process_bio_data(df):
    lawyers_bio = {}
    
    for _, row in df.iterrows():
        # Convert first and last names to string and handle NaN values
        first_name = str(row['First Name']).strip() if pd.notna(row['First Name']) else ""
        last_name = str(row['Last Name']).strip() if pd.notna(row['Last Name']) else ""
        
        full_name = f"{first_name} {last_name}".strip()
        
        # Skip empty names
        if not full_name:
            continue
        
        bio = {
            'level': str(row['Level/Title']) if pd.notna(row['Level/Title']) else "",
            'call': str(row['Call']) if pd.notna(row['Call']) else "",
            'jurisdiction': str(row['Jurisdiction']) if pd.notna(row['Jurisdiction']) else "",
            'location': str(row['Location']) if pd.notna(row['Location']) else "",
            'practice_areas': str(row['Area of Practise + Add Info']) if pd.notna(row['Area of Practise + Add Info']) else "",
            'industry_experience': str(row['Industry Experience']) if pd.notna(row['Industry Experience']) else "",
            'languages': str(row['Languages']) if pd.notna(row['Languages']) else "",
            'previous_in_house': str(row['Previous In-House Companies']) if pd.notna(row['Previous In-House Companies']) else "",
            'previous_firms': str(row['Previous Companies/Firms']) if pd.notna(row['Previous Companies/Firms']) else "",
            'education': str(row['Education']) if pd.notna(row['Education']) else "",
            'awards': str(row['Awards/Recognition']) if pd.notna(row['Awards/Recognition']) else "",
            'notable_items': str(row['Notable Items/Personal Details ']) if pd.notna(row['Notable Items/Personal Details ']) else "",
            'expert': str(row['Expert']) if pd.notna(row['Expert']) else ""
        }
        
        lawyers_bio[full_name] = bio
    
    return {
        'lawyers_bio': lawyers_bio
    }

# Function to combine skills and biographical data
def combine_lawyer_data(skills_data, bio_data):
    if not skills_data or not bio_data:
        return skills_data
    
    combined_lawyers = []
    
    for lawyer in skills_data['lawyers']:
        # Try to find matching biographical data
        name = lawyer['name']
        bio = None
        
        # Try exact match
        if name in bio_data['lawyers_bio']:
            bio = bio_data['lawyers_bio'][name]
        else:
            # Try partial match
            for bio_name, bio_info in bio_data['lawyers_bio'].items():
                # Check if first and last name parts match
                name_parts = name.lower().split()
                bio_name_parts = bio_name.lower().split()
                
                if any(part in bio_name.lower() for part in name_parts) and any(part in name.lower() for part in bio_name_parts):
                    bio = bio_info
                    break
        
        # Add biographical data if found
        if bio:
            lawyer['bio'] = bio
        else:
            lawyer['bio'] = {
                'level': '',
                'call': '',
                'jurisdiction': '',
                'location': '',
                'practice_areas': '',
                'industry_experience': '',
                'languages': '',
                'previous_in_house': '',
                'previous_firms': '',
                'education': '',
                'awards': '',
                'notable_items': '',
                'expert': ''
            }
        
        combined_lawyers.append(lawyer)
    
    return {
        'lawyers': combined_lawyers,
        'skill_map': skills_data['skill_map'],
        'unique_skills': skills_data['unique_skills']
    }

# Function to verify practice areas against biographical data
def verify_practice_areas(data):
    if not data:
        return data
    
    # Create a corrected version of the data
    corrected_data = {
        'lawyers': [],
        'skill_map': data['skill_map'],
        'unique_skills': data['unique_skills']
    }
    
    # Manual corrections for known issues
    manual_corrections = {
        'Nikki': 'Corporate',  # Corrected from Tax
        'Frank': 'Corporate',  # Corrected from IP
        # Add more corrections as needed
    }
    
    # Process each lawyer to update practice areas based on bio data
    for lawyer in data['lawyers']:
        lawyer_copy = lawyer.copy()  # Create a copy to modify
        
        # Check if lawyer's name contains any key in manual_corrections
        for name_part, correction in manual_corrections.items():
            if name_part in lawyer['name']:
                lawyer_copy['practice_area'] = correction
                break
        
        # If no manual correction and bio data is available, try to infer practice area
        if 'bio' in lawyer_copy and lawyer_copy['bio'].get('practice_areas'):
            practice_areas = lawyer_copy['bio']['practice_areas'].lower()
            
            # Simple mapping of bio practice areas to main categories
            if 'corporate' in practice_areas or 'commercial' in practice_areas or 'business' in practice_areas:
                lawyer_copy['practice_area'] = 'Corporate'
            elif 'litigation' in practice_areas or 'dispute' in practice_areas:
                lawyer_copy['practice_area'] = 'Litigation'
            elif 'intellectual property' in practice_areas or 'patent' in practice_areas or 'trademark' in practice_areas:
                lawyer_copy['practice_area'] = 'IP'
            elif 'employ' in practice_areas or 'labor' in practice_areas or 'labour' in practice_areas:
                lawyer_copy['practice_area'] = 'Employment'
            elif 'privacy' in practice_areas or 'data' in practice_areas:
                lawyer_copy['practice_area'] = 'Privacy'
            elif 'finance' in practice_areas or 'banking' in practice_areas:
                lawyer_copy['practice_area'] = 'Finance'
            elif 'real estate' in practice_areas or 'property' in practice_areas:
                lawyer_copy['practice_area'] = 'Real Estate'
            elif 'tax' in practice_areas:
                lawyer_copy['practice_area'] = 'Tax'
            # Keep original if no match
        
        corrected_data['lawyers'].append(lawyer_copy)
    
    return corrected_data

# Improved match_lawyers function that prioritizes biographical data
def match_lawyers(data, query, top_n=10):  # Changed from 5 to 10
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
        skill_score = 0
        matched_bio_reasons = []
        matched_skills = []
        all_criteria_matched = True if len(query_parts) > 1 else False
        
        # FIRST: Check biographical data for matches
        bio = lawyer.get('bio', {})
        
        # Create a single text string from all biographical data to search
        bio_text = " ".join([
            bio.get('practice_areas', ''),
            bio.get('expert', ''),
            bio.get('industry_experience', ''),
            bio.get('notable_items', ''),
            bio.get('previous_in_house', ''),
            bio.get('previous_firms', '')
        ]).lower()
        
        # Check each query part against biographical data
        for query_part in query_parts:
            part_matched_in_bio = False
            
            # Check for exact or partial matches in biographical data
            if query_part.strip() in bio_text:
                bio_score += 5  # High score for bio matches
                
                # Determine which bio field matched
                for field, value in bio.items():
                    if value and query_part.strip() in value.lower():
                        matched_bio_reasons.append({
                            'field': field,
                            'value': value
                        })
                        part_matched_in_bio = True
            
            # For multi-criteria queries, track if each part matched in bio
            if not part_matched_in_bio:
                for word in query_part.split():
                    if word in bio_text and len(word) > 3:  # Avoid matching small words
                        bio_score += 2
                        
                        # Determine which bio field matched
                        for field, value in bio.items():
                            if value and word in value.lower():
                                matched_bio_reasons.append({
                                    'field': field,
                                    'value': value
                                })
                                part_matched_in_bio = True
                                break
            
            # SECOND: Check skills data as a cross-reference
            part_matched_in_skills = False
            
            # Check each skill against the query part
            for skill, value in lawyer['skills'].items():
                skill_lower = skill.lower()
                
                # More precise matching - prefer exact matches over partial
                if skill_lower == query_part.strip():
                    # Exact match gets higher score
                    skill_score += value * 1.5
                    matched_skills.append({'skill': skill, 'value': value})
                    part_matched_in_skills = True
                elif query_part.strip() in skill_lower:
                    # Contains match
                    skill_score += value
                    matched_skills.append({'skill': skill, 'value': value})
                    part_matched_in_skills = True
                elif any(word in skill_lower for word in query_part.split() if len(word) > 3):
                    # Word match (for words > 3 chars to avoid matching small words)
                    skill_score += value * 0.5
                    matched_skills.append({'skill': skill, 'value': value})
                    part_matched_in_skills = True
            
            # For multi-criteria queries, check if this part matched in either bio or skills
            if len(query_parts) > 1 and not (part_matched_in_bio or part_matched_in_skills):
                all_criteria_matched = False
        
        # For multi-criteria queries, if not all criteria matched, reset scores
        if len(query_parts) > 1 and not all_criteria_matched:
            bio_score = 0
            skill_score = 0
            matched_bio_reasons = []
            matched_skills = []
        
        # Calculate final score with bio_score weighted higher
        final_score = (bio_score * 2) + skill_score
        
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
            
            matches.append({
                'lawyer': lawyer,
                'score': final_score,
                'bio_score': bio_score,
                'skill_score': skill_score,
                'matched_bio_reasons': list(unique_bio_reasons.values()),
                'matched_skills': sorted(list(unique_skills.values()), key=lambda x: x['value'], reverse=True)[:5]
            })
    
    # Sort by final score and take top N
    return sorted(matches, key=lambda x: x['score'], reverse=True)[:top_n]

# Updated function to format Claude's analysis prompt with clearer structure
def format_claude_prompt(query, matches):
    prompt = f"""
I need to analyze and provide detailed reasoning for why specific lawyers match a client's legal needs based on their expertise, skills, and background.

Client's Legal Need: "{query}"

I'll provide you with information about each matching lawyer. For each lawyer, please provide a detailed, specific explanation of why they would be an excellent match for this client need. Focus on their biographical information, expertise, and experience.

"""
    
    for i, match in enumerate(matches, 1):
        lawyer = match['lawyer']
        skills = match.get('matched_skills', [])
        bio_reasons = match.get('matched_bio_reasons', [])
        bio = lawyer.get('bio', {})
        
        prompt += f"LAWYER {i}: {lawyer['name']}\n"
        prompt += "---------------------------------------------\n"
        
        # Add matched biographical information FIRST - this is what matched in the search
        if bio_reasons:
            prompt += "MATCHING BIOGRAPHICAL FACTORS:\n"
            for reason in bio_reasons:
                field_name = reason['field'].replace('_', ' ').title()
                prompt += f"- {field_name}: {reason['value']}\n"
            prompt += "\n"
        
        # Add full biographical information
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
        prompt += "\n"
            
        # Add skills information as supporting evidence
        if skills:
            prompt += "SELF-REPORTED SKILLS:\n"
            for skill in skills:
                prompt += f"- {skill['skill']}: {skill['value']} points\n"
        
        prompt += "\n\n"
    
    prompt += """
For each lawyer, write ONE detailed paragraph (5-7 sentences) explaining why they are an excellent match for this client's needs.

Your explanation should:
1. Highlight relevant biographical details that match the client's needs
2. Mention specific practice areas, industry experience, or previous roles that are relevant
3. Include educational background if relevant
4. Reference their self-reported skills that support their expertise
5. Be specific and substantive - avoid generic language

Format your response as a JSON object where each key is the lawyer's name and each value is your explanation paragraph:

{
    "Lawyer Name 1": "Detailed explanation paragraph for this lawyer...",
    "Lawyer Name 2": "Detailed explanation paragraph for this lawyer...",
    "Lawyer Name 3": "Detailed explanation paragraph for this lawyer..."
}

DO NOT include any additional text outside of this JSON structure.
"""
    return prompt

# Improved function to generate more detailed, personalized fallback explanations
def generate_fallback_explanation(lawyer, bio, skills):
    """
    Generate a detailed, personalized explanation for why a lawyer matches the client's needs.
    Focuses on their biographical information first, then corroborates with self-reported skills.
    """
    # Extract key information
    name = lawyer['name']
    practice_areas = bio.get('practice_areas', '')
    industry_exp = bio.get('industry_experience', '')
    previous_exp = bio.get('previous_in_house', '')
    previous_firms = bio.get('previous_firms', '')
    education = bio.get('education', '')
    expertise = bio.get('expert', '')
    level = bio.get('level', '')
    call = bio.get('call', '')
    jurisdiction = bio.get('jurisdiction', '')
    
    # Build a narrative explanation focused on biographical data first
    explanation_parts = []
    
    # Start with their level and experience
    if level or call:
        experience_intro = f"{name} "
        if level:
            experience_intro += f"is a {level} "
        if call:
            years_exp = 2025 - int(call.split('(')[1].split(')')[0]) if '(' in call and ')' in call else None
            if years_exp:
                experience_intro += f"with {years_exp} years of legal experience "
        
        if practice_areas:
            experience_intro += f"specializing in {practice_areas}"
        
        explanation_parts.append(experience_intro.strip() + ".")
    
    # Add information about their expertise areas as they relate to the search query
    if practice_areas:
        practice_areas_part = f"Their extensive experience in {practice_areas} "
        if industry_exp:
            practice_areas_part += f"with specific focus on the {industry_exp} sectors "
        practice_areas_part += "provides the exact expertise needed for this client matter."
        explanation_parts.append(practice_areas_part)
    
    # Add background information that shows breadth of experience
    if previous_exp or previous_firms:
        background_part = "Their background "
        if previous_exp and previous_firms:
            background_part += f"includes both in-house experience at {previous_exp} and professional work with {previous_firms}, "
        elif previous_exp:
            background_part += f"includes valuable in-house experience at {previous_exp}, "
        elif previous_firms:
            background_part += f"at {previous_firms} "
        
        background_part += "gives them practical insights into both the legal and business aspects of this matter."
        explanation_parts.append(background_part)
    
    # Add education information
    if education:
        explanation_parts.append(f"Their education from {education} provides a strong theoretical foundation for handling this client's needs.")
    
    # Add jurisdiction information if relevant
    if jurisdiction:
        explanation_parts.append(f"Their qualification to practice in {jurisdiction} is particularly valuable for this client matter.")
    
    # Connect biographical information with self-reported skills as confirmation
    if skills:
        skill_names = [f"{s['skill']} ({s['value']})" for s in skills[:3]]
        skills_part = f"Their self-reported expertise ratings in {', '.join(skill_names)} confirms their confidence and capability in precisely the areas needed for this matter."
        explanation_parts.append(skills_part)
    
    # Add expertise statement as a closing point if available
    if expertise:
        explanation_parts.append(f"Their recognized expertise in {expertise} makes them an outstanding choice for this client.")
    
    # Combine parts into a complete paragraph
    # If we have too few parts, add a generic sentence
    if len(explanation_parts) < 3:
        explanation_parts.append(f"{name} has the perfect combination of experience, expertise, and qualifications to effectively address this client's specific legal needs.")
    
    return " ".join(explanation_parts)

# Function to call Claude API using requests instead of anthropic client
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
        st.warning("API key not found. Using mock reasoning data instead.")
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
            "max_tokens": 1000,
            "temperature": 0.2,
            "system": "You are a legal resource coordinator that analyzes lawyer expertise matches. You provide detailed, factual explanations about why specific lawyers match particular client legal needs based on their biographical information and self-reported skills. Be specific and thorough in your analysis, highlighting the exact qualifications that make each lawyer a good match. Focus primarily on biographical data (practice areas, experience, education) and use self-reported skills as supporting evidence.",
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
            st.warning("Could not parse the JSON response from Claude. Using fallback explanations.")
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

# Fallback function for generating mock lawyer reasoning
def call_claude_api_fallback(matches):
    try:
        lawyer_explanations = {}
        
        for match in matches:
            lawyer = match['lawyer']
            bio = lawyer.get('bio', {})
            skills = match.get('matched_skills', [])
            
            # Generate explanation using the helper function
            lawyer_explanations[lawyer['name']] = generate_fallback_explanation(lawyer, bio, skills)
        
        return lawyer_explanations
    except Exception as e:
        return {"error": f"Could not generate fallback explanations: {str(e)}"}

# Enhanced feedback function with submit button
def add_realtime_feedback():
    st.markdown("---")
    st.markdown('<div class="feedback-container">', unsafe_allow_html=True)
    st.markdown('<div class="feedback-title">Provide Feedback on Search Results</div>', unsafe_allow_html=True)
    
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
    
    # Create a text area for feedback
    feedback = st.text_area(
        "Additional feedback on search results:",
        value=st.session_state.get('feedback_text', ''),
        key="feedback_input",
        height=100,
        placeholder="Please share your thoughts on why these results are helpful or what could be improved..."
    )
    
    # Update the feedback text in session state
    st.session_state.feedback_text = feedback
    
    # Create a submit button for feedback
    submit_col1, submit_col2 = st.columns([3, 1])
    with submit_col2:
        submit_button = st.button("📝 Submit Feedback", key="submit_feedback", type="primary", use_container_width=True)
    
    # Handle submit button click
    if submit_button:
        if not feedback and not expected_lawyers and not incorrect_lawyers and rating == "":
            st.warning("Please provide at least some feedback before submitting.")
        else:
            # Create feedback data object
            feedback_data = {
                "query": st.session_state.get('query', ''),
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "feedback": feedback,
                "user_email": user_email if user_email else 'anonymous',
                "user_role": user_role,
                "lawyer_results": json.dumps([m['lawyer']['name'] for m in matches]) if 'matches' in globals() else "[]",
                "expected_lawyers": expected_lawyers,
                "rating": rating,
                "incorrect_lawyers": incorrect_lawyers
            }
            
            # Save to Supabase
            result = save_feedback_to_supabase(feedback_data)
            
            if result['success']:
                st.session_state.feedback_saved = True
                st.session_state.last_saved_feedback = feedback
                st.session_state.last_feedback_id = result.get('id')
                st.success("✓ Feedback submitted successfully! Thank you for helping us improve the matching system.")
            else:
                st.session_state.feedback_saved = False
                st.session_state.feedback_error = result.get('message')
                st.error(f"Error saving feedback: {result.get('message')}")
    
    # If feedback was already saved through other means
    elif st.session_state.get('feedback_saved', False) and st.session_state.get('last_saved_feedback', '') == feedback:
        st.success("✓ Feedback already saved. Thank you for your input!")
    
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
st.markdown("Match client legal needs with the right lawyer based on expertise")

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
        with st.spinner("Matching client needs with our legal experts..."):
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
                st.markdown(f"Found {len(matches)} lawyers matching client needs:")
                
                # Sort alphabetically for display (not by score)
                sorted_matches = sorted(matches, key=lambda x: x['lawyer']['name'])
                
                for match in sorted_matches:
                    lawyer = match['lawyer']
                    matched_skills = match['matched_skills']
                    matched_bio_reasons = match.get('matched_bio_reasons', [])
                    
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
                        
                        # Add matched bio reasons section
                        if matched_bio_reasons:
                            html_output += '<div style="margin-top: 10px;"><strong>Matched Biographical Factors:</strong><br/>'
                            for reason in matched_bio_reasons:
                                field_name = reason['field'].replace('_', ' ').title()
                                html_output += f'<span class="bio-match-tag">{field_name}</span> '
                            html_output += '</div>'
                        
                        # Add skill tags
                        html_output += f"""
                            <div style="margin-top: 10px;">
                                <strong>Relevant Expertise:</strong><br/>
                                {"".join([f'<span class="skill-tag">{skill["skill"]}: {skill["value"]}</span>' for skill in matched_skills])}
                            </div>
                        """
                        
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
                                # Generate a fallback explanation for this specific lawyer
                                lawyer_reasoning = generate_fallback_explanation(lawyer, bio, matched_skills)
                        else:
                            # If reasoning is not a dictionary, generate a fallback explanation
                            lawyer_reasoning = generate_fallback_explanation(lawyer, bio, matched_skills)
                        
                        # Make sure the reasoning text is properly escaped and formatted
                        # Ensure lawyer_reasoning is a string before using html.escape
                        if not isinstance(lawyer_reasoning, str):
                            lawyer_reasoning = str(lawyer_reasoning)
                            
                        # Escape any HTML tags that might be in the text
                        lawyer_reasoning_escaped = html.escape(lawyer_reasoning)
                        
                        # Add the reasoning section to the HTML output with proper formatting
                        html_output += f"""
                            <div class="reasoning-box">
                                <div class="match-rationale-title">WHY THIS LAWYER IS AN EXCELLENT MATCH:</div>
                                <p>{lawyer_reasoning_escaped}</p>
                            </div>
                        </div>
                        """
                        
                        # Render the HTML
                        st.markdown(html_output, unsafe_allow_html=True)
                
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
        st.markdown("### Most Common Legal Expertise Areas")
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
        Be as specific as possible about their requirements, including:
        
        - The type of legal expertise needed
        - Any industry-specific requirements
        - Geographic considerations (e.g., province-specific needs)
        - The nature of the legal matter
        - Timeframe and urgency
        
        For multi-criteria searches, use "AND" or "&" between criteria (e.g., "commercial contracts AND hospitality").
        The system will match the query with lawyers who have relevant biographical information and self-reported expertise in those areas.
        """)

# Footer
st.markdown("---")
st.markdown(
    "This internal tool uses biographical information and self-reported expertise from 84 lawyers who distributed 120 points across 167 different legal skills. "
    "Results are sorted alphabetically and matches are based on biographical data with self-reported skill points as supporting evidence. "
    "Last updated: April 18, 2025"
)
        
