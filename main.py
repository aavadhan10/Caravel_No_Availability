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

# Initialize Supabase client
def init_supabase():
    try:
        supabase_url = st.secrets["supabase"]["url"]
        supabase_key = st.secrets["supabase"]["key"]
        
        supabase: Client = create_client(supabase_url, supabase_key)
        return supabase
    except Exception as e:
        st.error(f"Error initializing Supabase: {e}")
        return None

# Initialize Supabase client
supabase = init_supabase()

# Function to save feedback to Supabase
def save_feedback_to_supabase(feedback_data):
    try:
        # Insert feedback into 'feedback' table
        response = supabase.table('feedback').insert(feedback_data).execute()
        return {"success": True, "message": "Feedback saved to database", "id": response.data[0].get('id') if response.data else None}
    except Exception as e:
        st.error(f"Error saving feedback: {e}")
        return {"success": False, "message": str(e)}

# Function to get feedback from Supabase
def get_feedback_from_supabase(limit=50):
    try:
        # Get feedback from 'feedback' table
        response = supabase.table('feedback').select('*').order('created_at', desc=True).limit(limit).execute()
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
</style>
""", unsafe_allow_html=True)

# Set up sidebar
st.sidebar.title("⚖️ Legal Expert Finder")
st.sidebar.title("About")
st.sidebar.info(
    "This internal tool helps match client legal needs with the right lawyer based on self-reported expertise. "
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
    "Commercial lease agreement review"
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
        
        return combined_data
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

# Updated match_lawyers function with improved matching logic
def match_lawyers(data, query, top_n=10):  # Changed from 5 to 10
    if not data:
        return []
    
    # Convert query to lowercase for case-insensitive matching
    lower_query = query.lower()
    
    # Test users to exclude
    excluded_users = ["Ankita", "Test", "Tania"]
    
    # For M&A queries, expand the search terms
    if "m&a" in lower_query.lower() or "merger" in lower_query.lower() or "acquisition" in lower_query.lower():
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
        # Skip test users
        if any(excluded_name in lawyer['name'] for excluded_name in excluded_users):
            continue
            
        score = 0
        matched_skills = []
        all_criteria_matched = True if len(query_parts) > 1 else False
        
        # Check each query part
        for query_part in query_parts:
            part_matched = False
            
            # Check each skill against the query part
            for skill, value in lawyer['skills'].items():
                skill_lower = skill.lower()
                
                # More precise matching - prefer exact matches over partial
                if skill_lower == query_part.strip():
                    # Exact match gets higher score
                    score += value * 2
                    matched_skills.append({'skill': skill, 'value': value})
                    part_matched = True
                elif query_part.strip() in skill_lower:
                    # Contains match
                    score += value * 1.5
                    matched_skills.append({'skill': skill, 'value': value})
                    part_matched = True
                elif any(word in skill_lower for word in query_part.split()):
                    # Word match
                    score += value
                    matched_skills.append({'skill': skill, 'value': value})
                    part_matched = True
            
            # For multi-criteria queries, track if each part matched
            if len(query_parts) > 1 and not part_matched:
                all_criteria_matched = False
        
        # For multi-criteria queries, if not all criteria matched, reset score
        if len(query_parts) > 1 and not all_criteria_matched:
            score = 0
            matched_skills = []
        
        # Add lawyer to matches if scored
        if score > 0:
            # Remove duplicate skills
            unique_skills = {}
            for skill in matched_skills:
                skill_name = skill['skill']
                if skill_name not in unique_skills or skill['value'] > unique_skills[skill_name]['value']:
                    unique_skills[skill_name] = skill
            
            matches.append({
                'lawyer': lawyer,
                'score': score,
                'matched_skills': sorted(list(unique_skills.values()), key=lambda x: x['value'], reverse=True)[:5]
            })
    
    # Sort by score and take top N
    return sorted(matches, key=lambda x: x['score'], reverse=True)[:top_n]  # Changed from 5 to 10

# Function to format Claude's analysis prompt
def format_claude_prompt(query, matches):
    prompt = f"""
I need to analyze and provide detailed reasoning for why specific lawyers match a client's legal needs based on their expertise, skills, and background.

Client's Legal Need: "{query}"

Here are the matching lawyers with their skills and biographical information:

"""
    
    for i, match in enumerate(matches, 1):
        lawyer = match['lawyer']
        skills = match['matched_skills']
        bio = lawyer['bio']
        
        prompt += f"LAWYER {i}: {lawyer['name']}\n"
        prompt += "---------------------------------------------\n"
        
        # Add skills information
        prompt += "RELEVANT SKILLS:\n"
        for skill in skills:
            prompt += f"- {skill['skill']}: {skill['value']} points\n"
        
        # Add biographical information
        prompt += "\nBIOGRAPHICAL INFORMATION:\n"
        if bio['level']:
            prompt += f"- Level/Title: {bio['level']}\n"
        if bio['call']:
            prompt += f"- Called to Bar: {bio['call']}\n"
        if bio['jurisdiction']:
            prompt += f"- Jurisdiction: {bio['jurisdiction']}\n"
        if bio['location']:
            prompt += f"- Location: {bio['location']}\n"
        if bio['practice_areas']:
            prompt += f"- Practice Areas: {bio['practice_areas']}\n"
        if bio['industry_experience']:
            prompt += f"- Industry Experience: {bio['industry_experience']}\n"
        if bio['previous_in_house']:
            prompt += f"- Previous In-House Experience: {bio['previous_in_house']}\n"
        if bio['previous_firms']:
            prompt += f"- Previous Law Firms: {bio['previous_firms']}\n"
        if bio['education']:
            prompt += f"- Education: {bio['education']}\n"
        if bio['awards']:
            prompt += f"- Awards/Recognition: {bio['awards']}\n"
        if bio['expert']:
            prompt += f"- Areas of Expertise: {bio['expert']}\n"
        if bio['notable_items']:
            prompt += f"- Notable Experience: {bio['notable_items']}\n"
            
        prompt += "\n\n"
    
    prompt += """
For each lawyer, provide a DETAILED explanation (at least 4-5 sentences) of why they would be an excellent match for this client need. Include specific aspects of their:

1. Skills relevance - How their specific skills directly relate to the client's needs
2. Industry background - How their industry experience aligns with the client's requirements
3. Prior experience - Highlight relevant previous work or clients that demonstrate fit
4. Education or certifications that may be valuable for this matter
5. Geographic or jurisdictional advantages if relevant

Your analysis should be thorough and specific, referencing their unique qualifications rather than generic statements. Format your response as a detailed paragraph for each lawyer that explains exactly why they're well-positioned to address this specific client need.

Format your response in JSON like this:
{
    "lawyer1_name": "Detailed explanation of why lawyer 1 is an excellent match...",
    "lawyer2_name": "Detailed explanation of why lawyer 2 is an excellent match...",
    "lawyer3_name": "Detailed explanation of why lawyer 3 is an excellent match..."
}
"""
    return prompt

# Function to call Claude API using requests instead of anthropic client
def call_claude_api(prompt):
    api_key = st.secrets.get("anthropic", {}).get("api_key", "YOUR_API_KEY_HERE")
    
    # Handle the case where no API key is provided
    if api_key == "YOUR_API_KEY_HERE":
        # Return mock reasoning data for the lawyers
        try:
            return {
                match['lawyer']['name']: f"This lawyer has strong expertise in {', '.join([s['skill'] for s in match['matched_skills'][:2]])}, making them highly qualified for this client matter. Their background in {match['lawyer']['bio'].get('practice_areas', 'relevant practice areas')} directly aligns with the client's requirements. With experience at {match['lawyer']['bio'].get('previous_in_house', 'relevant organizations')} and {match['lawyer']['bio'].get('previous_firms', 'law firms')}, they bring practical industry knowledge to the table. Their education from {match['lawyer']['bio'].get('education', 'respected institutions')} provides additional theoretical foundation for handling this matter effectively. They've dedicated significant points to these skills in their self-assessment, indicating deep confidence and capability in these areas."
                for match in matches[:10]  # Updated to include up to 10 matches
            }
        except NameError:
            # Fallback if 'matches' is not defined in this scope
            return {"Error": "No API key provided and could not generate mock data"}
    
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
        
        # Request payload
        payload = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 1000,
            "temperature": 0.0,
            "system": "You are a legal resource coordinator that analyzes lawyer expertise matches. You provide detailed, factual explanations about why specific lawyers match particular client legal needs based on their self-reported skills and biographical information. Be specific and thorough in your analysis, highlighting the exact qualifications that make each lawyer a good match.",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        # Make the request
        response = requests.post(url, headers=headers, json=payload)
        
        # Check for successful response
        if response.status_code == 200:
            response_json = response.json()
            response_text = response_json.get("content", [{}])[0].get("text", "")
            
            # Find JSON part in the response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                return {"error": "Could not extract JSON from Claude's response"}
        else:
            return {"error": f"API call failed with status code {response.status_code}: {response.text}"}
            
    except Exception as e:
        st.error(f"Error calling Claude API: {str(e)}")
        
        # Provide a more detailed error message to help debugging
        import traceback
        st.error(f"Error details: {traceback.format_exc()}")
        
        # Return a fallback response
        return {"error": f"API error: {str(e)}"}

# Real-time feedback function with Supabase integration
def add_realtime_feedback():
    st.markdown("---")
    st.markdown('<div class="feedback-container">', unsafe_allow_html=True)
    st.markdown('<div class="feedback-title">Provide Feedback</div>', unsafe_allow_html=True)
    
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
                "user_role": "partner",
                "lawyer_results": json.dumps([m['lawyer']['name'] for m in matches]) if 'matches' in globals() else "[]"
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
    
    # Optional email field for feedback attribution
    user_email = st.text_input(
        "Your email (optional):",
        value=st.session_state.get('user_email', ''),
        key="user_email_input",
        placeholder="Enter your email to associate with feedback"
    )
    
    # Store email in session state
    st.session_state.user_email = user_email
    
    # Create a text area that calls the callback when changed
    feedback = st.text_area(
        "Help us improve our lawyer matching system:",
        value=st.session_state.feedback_text,
        key="feedback_input",
        on_change=on_feedback_change,
        height=100,
        placeholder="Please share your thoughts on the search results or suggest improvements..."
    )
    
    # Update the feedback text in session state
    st.session_state.feedback_text = feedback
    
    # If there's feedback, show it's being saved in real-time
    if feedback:
        # If feedback saved successfully
        if st.session_state.get('feedback_saved', False):
            # Display a success message
            st.success("✓ Feedback saved automatically")
    
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
    else:
        st.warning("Feedback data format is not as expected.")

# Add an admin mode toggle to the sidebar
def add_admin_mode():
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Admin Options")
    
    # Admin login
    admin_password = st.sidebar.text_input("Admin Password", type="password")
    
    if admin_password == st.secrets.get("supabase", {}).get("admin_password", "admin123"):
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
        "M&A",  # Updated from "M&A without tech companies"
        "Privacy compliance",
        "Startup law",
        "Employment issues",
        "Intellectual property protection",
        "Employment termination reviews",
        "Incorporation and corporate record keeping",
        "Healthcare compliance regulations",
        "Fintech regulatory compliance",
        "Service agreements, including SaaS contracts"
    ]

    # Query input section
    query = st.text_area(
        "Describe client's legal needs in detail:", 
        value=st.session_state['query'],
        height=100,
        placeholder="Example: Client needs a lawyer with blockchain governance experience for cross-border cryptocurrency transactions",
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
                
                # Display results
                st.markdown("## Matching Legal Experts")
                st.markdown(f"Found {len(matches)} lawyers matching client needs:")
                
                # Sort alphabetically for display (not by score)
                sorted_matches = sorted(matches, key=lambda x: x['lawyer']['name'])
                
                for match in sorted_matches:
                    lawyer = match['lawyer']
                    matched_skills = match['matched_skills']
                    
                    with st.container():
                        # Determine style classes
                        
                        # Get bio data
                        bio = lawyer['bio'] if 'bio' in lawyer else {}
                        
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
                        
                        # Add the rest of the card
                        html_output += f"""
                            <div style="margin-top: 10px;">
                                <strong>Relevant Expertise:</strong><br/>
                                {"".join([f'<span class="skill-tag">{skill["skill"]}: {skill["value"]}</span>' for skill in matched_skills])}
                            </div>
                            <div class="reasoning-box">
                                <div class="match-rationale-title">WHY THIS LAWYER IS AN EXCELLENT MATCH:</div>
                                {reasoning.get(lawyer['name'], 'This lawyer has relevant expertise in the areas described in the client query.')}
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
            
            # Quick stats
            st.markdown("### Firm Resource Overview")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Lawyers", len(data['lawyers']))
            with col2:
                st.metric("Expertise Areas", len(data['unique_skills']))
            
            st.markdown("### Instructions for Matching")
            st.markdown("""
            Enter your client's specific legal needs above or select a common query to find matching legal experts. 
            Be as specific as possible about their requirements, including:
            
            - The type of legal expertise needed
            - Any industry-specific requirements
            - Geographic considerations (e.g., province-specific needs)
            - The nature of the legal matter
            - Timeframe and urgency
            
            The system will match the query with lawyers who have self-reported expertise in those areas.
            """)

    # Footer
    st.markdown("---")
    st.markdown(
        "This internal tool uses self-reported expertise from 84 lawyers who distributed 120 points across 167 different legal skills. "
        "Results are sorted alphabetically and matches are based on keyword relevance and self-reported skill points. "
        "Last updated: April 12, 2025"
    )
