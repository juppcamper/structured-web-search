import streamlit as st
import os
import requests
from dotenv import load_dotenv
from firecrawl import FirecrawlApp, ScrapeOptions
import json
from openai import OpenAI
import pandas as pd
import time
from datetime import datetime
import csv
from pathlib import Path

# Load environment variables
load_dotenv()

# Create output directory if it doesn't exist
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="Firecrawl Web Scraper",
    page_icon="🌐",
    layout="wide"
)

# Title and description
st.title("🌐 Firecrawl Web Scraper")
st.markdown("Scrape websites and convert them to LLM-ready markdown using the Firecrawl API")

# Initialize Firecrawl app
@st.cache_resource
def get_firecrawl_app():
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        st.error("❌ FIRECRAWL_API_KEY not found in .env file")
        return None
    return FirecrawlApp(api_key=api_key)

# Generate JSON schema from field definitions
def generate_json_schema(fields):
    properties = {}
    required = []
    
    for field in fields:
        field_name = field['name']
        field_type = field['type']
        field_desc = field['description']
        is_required = field['required']
        
        if field_type == "string":
            properties[field_name] = {
                "type": "string",
                "description": field_desc
            }
        elif field_type == "number":
            properties[field_name] = {
                "type": "number",
                "description": field_desc
            }
        elif field_type == "boolean":
            properties[field_name] = {
                "type": "boolean",
                "description": field_desc
            }
        elif field_type == "array":
            properties[field_name] = {
                "type": "array",
                "items": {"type": "string"},
                "description": field_desc
            }
        elif field_type == "object":
            properties[field_name] = {
                "type": "object",
                "description": field_desc
            }
        
        if is_required:
            required.append(field_name)
    
    schema = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False
    }
    
    return schema

# Generate structured prompt for both providers
def generate_structured_prompt(original_query, fields):
    field_descriptions = []
    for field in fields:
        required_text = " (REQUIRED)" if field['required'] else " (optional)"
        field_descriptions.append(f"  \"{field['name']}\": {field['type']} - {field['description']}{required_text}")
    
    structured_prompt = f"""{original_query}

CRITICAL INSTRUCTIONS FOR RESPONSE FORMAT:
- You MUST respond with ONLY valid JSON data
- Do NOT include any explanatory text, introductions, or conclusions
- Do NOT use markdown code blocks or formatting
- Do NOT add any text before or after the JSON
- Your entire response must be parseable as JSON

Required JSON structure:
{{
{chr(10).join(field_descriptions)}
}}

RESPOND WITH ONLY THE JSON OBJECT - NO OTHER TEXT WHATSOEVER."""
    
    return structured_prompt

# OpenAI web search function
def research_with_openai(query, model="gpt-4.1", structured_fields=None):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ OPENAI_API_KEY not found in .env file")
        return None
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Modify query for structured output if requested
        final_query = query
        if structured_fields:
            final_query = generate_structured_prompt(query, structured_fields)
        
        response = client.responses.create(
            model=model,
            tools=[{"type": "web_search_preview"}],
            input=final_query
        )
        
        # Extract the response content
        if hasattr(response, 'output_text'):
            content = response.output_text
            
            # Try to parse as JSON if structured output was requested
            structured_data = None
            if structured_fields:
                try:
                    # First try to parse the entire response as JSON
                    structured_data = json.loads(content.strip())
                except json.JSONDecodeError:
                    try:
                        # Look for JSON in the response using improved regex
                        import re
                        # Find JSON objects (handles nested braces better)
                        json_pattern = r'\{(?:[^{}]|{[^{}]*})*\}'
                        json_matches = re.findall(json_pattern, content, re.DOTALL)
                        if json_matches:
                            # Try to parse the largest JSON match (likely the most complete)
                            largest_match = max(json_matches, key=len)
                            structured_data = json.loads(largest_match)
                        else:
                            # Last resort: try to find any JSON-like content
                            lines = content.strip().split('\n')
                            for line in lines:
                                line = line.strip()
                                if line.startswith('{') and line.endswith('}'):
                                    structured_data = json.loads(line)
                                    break
                    except (json.JSONDecodeError, ValueError):
                        st.warning(f"⚠️ Could not parse structured data from OpenAI response. Raw content: {content[:200]}...")
            
            return {
                "provider": "openai",
                "content": content,
                "structured_data": structured_data,
                "raw_response": response
            }
        else:
            st.error("❌ Unexpected OpenAI response format")
            return None
            
    except Exception as e:
        st.error(f"❌ Error calling OpenAI API: {str(e)}")
        return None

# Perplexity AI research function
def research_with_perplexity(query, model="sonar-pro", structured_fields=None):
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        st.error("❌ PERPLEXITY_API_KEY not found in .env file")
        return None
    
    try:
        # Modify query for structured output if requested
        final_query = query
        if structured_fields:
            final_query = generate_structured_prompt(query, structured_fields)
        
        response = requests.post(
            'https://api.perplexity.ai/chat/completions',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            },
            json={
                'model': model,
                'messages': [
                    {
                        'role': 'user',
                        'content': final_query
                    }
                ]
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content'] if 'choices' in result else ""
            
            # Try to extract structured data if requested
            structured_data = None
            if structured_fields:
                try:
                    # First try to parse the entire response as JSON
                    structured_data = json.loads(content.strip())
                except json.JSONDecodeError:
                    try:
                        # Look for JSON in the response using improved regex
                        import re
                        # Find JSON objects (handles nested braces better)
                        json_pattern = r'\{(?:[^{}]|{[^{}]*})*\}'
                        json_matches = re.findall(json_pattern, content, re.DOTALL)
                        if json_matches:
                            # Try to parse the largest JSON match (likely the most complete)
                            largest_match = max(json_matches, key=len)
                            structured_data = json.loads(largest_match)
                        else:
                            # Last resort: try to find any JSON-like content
                            lines = content.strip().split('\n')
                            for line in lines:
                                line = line.strip()
                                if line.startswith('{') and line.endswith('}'):
                                    structured_data = json.loads(line)
                                    break
                    except (json.JSONDecodeError, ValueError):
                        st.warning(f"⚠️ Could not parse structured data from Perplexity response. Raw content: {content[:200]}...")
            
            return {
                "provider": "perplexity",
                "content": content,
                "structured_data": structured_data,
                "raw_response": result
            }
        else:
            st.error(f"❌ Perplexity API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"❌ Error calling Perplexity API: {str(e)}")
        return None

def clean_result_for_json(result_data):
    """Clean result data to make it JSON serializable"""
    if not isinstance(result_data, dict):
        return str(result_data)
    
    cleaned_data = {}
    for key, value in result_data.items():
        if key == 'raw_response':
            # Skip raw_response as it contains non-serializable objects
            continue
        elif isinstance(value, dict):
            cleaned_data[key] = clean_result_for_json(value)
        elif isinstance(value, list):
            cleaned_data[key] = [clean_result_for_json(item) if isinstance(item, dict) else str(item) for item in value]
        else:
            # Convert other types to string to ensure serializability
            try:
                json.dumps(value)  # Test if it's JSON serializable
                cleaned_data[key] = value
            except (TypeError, ValueError):
                cleaned_data[key] = str(value)
    
    return cleaned_data

def save_batch_result(batch_id, row_name, result_data, query=None):
    """Save individual batch result to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create batch-specific subfolder
    batch_folder = OUTPUT_DIR / batch_id
    batch_folder.mkdir(exist_ok=True)
    
    filename = f"{row_name}_{timestamp}.json"
    filepath = batch_folder / filename
    
    # Clean the result data to make it JSON serializable
    cleaned_result = clean_result_for_json(result_data)
    
    save_data = {
        'batch_id': batch_id,
        'row_name': row_name,
        'timestamp': timestamp,
        'result': cleaned_result
    }
    
    if query:
        save_data['query'] = query
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    return filepath

def load_batch_results(batch_id):
    """Load all results for a specific batch"""
    results = []
    batch_folder = OUTPUT_DIR / batch_id
    
    if not batch_folder.exists():
        return results
    
    for filepath in batch_folder.glob("*.json"):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                results.append(json.load(f))
        except Exception as e:
            st.warning(f"Could not load {filepath}: {e}")
    return results

def create_batch_summary(results):
    """Create a summary table from batch results"""
    summary_data = []
    for result in results:
        row_name = result.get('row_name', 'Unknown')
        row_index = result.get('row_index', 'Unknown')
        timestamp = result.get('timestamp', 'Unknown')
        
        # Extract structured data if available
        structured_data = result.get('result', {}).get('structured_data', {})
        if structured_data:
            # Handle both list and dictionary structured data
            if isinstance(structured_data, list):
                # If it's a list, expand each item into a separate row
                for item in structured_data:
                    if isinstance(item, dict):
                        # Create a row for each person/item in the list
                        row = {
                            'row_index': row_index,
                            'row_name': row_name,
                            'timestamp': timestamp,
                            **item  # Unpack all the person's data (vorname, nachname, position, stadt, etc.)
                        }
                        summary_data.append(row)
                    else:
                        # Fallback for non-dict items in list
                        row = {
                            'row_index': row_index,
                            'row_name': row_name,
                            'timestamp': timestamp,
                            'item_data': str(item)
                        }
                        summary_data.append(row)
            elif isinstance(structured_data, dict):
                # If it's a dictionary, unpack it into the row
                row = {
                    'row_index': row_index,
                    'row_name': row_name,
                    'timestamp': timestamp,
                    **structured_data
                }
                summary_data.append(row)
            else:
                # Fallback for other types
                row = {
                    'row_index': row_index,
                    'row_name': row_name,
                    'timestamp': timestamp,
                    'structured_data': str(structured_data)
                }
                summary_data.append(row)
        else:
            # Fallback to basic info
            content = result.get('result', {}).get('content', '')
            row = {
                'row_index': row_index,
                'row_name': row_name,
                'timestamp': timestamp,
                'content_preview': content[:200] + '...' if len(content) > 200 else content
            }
            summary_data.append(row)
    
    return pd.DataFrame(summary_data)

def process_batch_cities(row_indices, research_query_template, search_provider, model, structured_fields=None, delay=1, csv_data=None, column_mappings=None):
    """Process a batch of rows with research queries"""
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, row_idx in enumerate(row_indices):
        if row_idx not in csv_data.index:
            continue
            
        row = csv_data.loc[row_idx]
        status_text.text(f"Processing row {row_idx} ({i+1}/{len(row_indices)})")
        
        # Create row-specific query with all mapped variables
        query = research_query_template
        if csv_data is not None and column_mappings:
            # Replace all variable placeholders with actual values
            for var, col in column_mappings.items():
                placeholder = f"{{{var}}}"
                if placeholder in query:
                    value = str(row[col]) if not pd.isna(row[col]) else ""
                    query = query.replace(placeholder, value)
        
        # Create a descriptive name for this row (for display and file naming)
        row_name = f"row_{row_idx}"
        if column_mappings:
            # Try to create a more descriptive name using the first mapped variable
            first_var = list(column_mappings.keys())[0] if column_mappings else None
            if first_var:
                first_col = column_mappings[first_var]
                if first_col in row and not pd.isna(row[first_col]):
                    row_name = str(row[first_col])
        
        try:
            # Make API call based on provider
            if search_provider == "OpenAI":
                result = research_with_openai(query, model, structured_fields)
            else:
                result = research_with_perplexity(query, model, structured_fields)
            
            if result:
                # Save individual result
                save_batch_result(batch_id, row_name, result, query)
                results.append({
                    'row_index': row_idx,
                    'row_name': row_name,
                    'result': result,
                    'query': query,
                    'timestamp': datetime.now().isoformat()
                })
                st.success(f"✅ Row {row_idx} ({row_name}) completed")
            else:
                st.error(f"❌ Row {row_idx} ({row_name}) failed")
                
        except Exception as e:
            st.error(f"❌ Error processing row {row_idx} ({row_name}): {str(e)}")
        
        # Update progress
        progress_bar.progress((i + 1) / len(row_indices))
        
        # Add delay between requests to avoid rate limiting
        if i < len(row_indices) - 1:  # Don't delay after the last request
            time.sleep(delay)
    
    progress_bar.empty()
    status_text.empty()
    
    return batch_id, results

# Sidebar for options
with st.sidebar:
    st.header("⚙️ Options")
    
    # Processing mode selection
    processing_mode = st.radio(
        "Mode:",
        ["Single Request", "Batch Processing"],
        help="Single: Process one URL/query | Batch: Process multiple cities from CSV"
    )
    
    if processing_mode == "Batch Processing":
        st.subheader("📊 Batch Options")
        
        # Web search provider selection for batch
        batch_search_provider = st.selectbox(
            "Search Provider:",
            ["OpenAI", "Perplexity"],
            help="Choose the web search provider for batch processing"
        )
        
        if batch_search_provider == "OpenAI":
            batch_openai_model = st.selectbox(
                "OpenAI Model:",
                ["gpt-4.1", "o4-mini"],
                help="Choose the OpenAI model for web search"
            )
        else:
            batch_perplexity_model = st.selectbox(
                "Perplexity Model:",
                ["sonar-pro", "sonar-medium-online", "llama-3.1-sonar-small-128k-online"],
                help="Choose the Perplexity AI model for research"
            )
        
        # Delay between requests
        request_delay = st.number_input(
            "Delay between requests (seconds):",
            min_value=0.5,
            max_value=10.0,
            value=1.0,
            step=0.5,
            help="Delay between API calls to avoid rate limiting"
        )
        
        # Load previous batch results
        st.subheader("📂 Load Previous Results")
        
        # Get list of batch IDs from output directory (subfolders)
        batch_folders = [folder for folder in OUTPUT_DIR.iterdir() if folder.is_dir()]
        batch_ids = [folder.name for folder in batch_folders]
        
        if batch_ids:
            selected_batch = st.selectbox(
                "Select batch to load:",
                sorted(batch_ids, reverse=True),
                help="Load results from a previous batch processing session"
            )
            
            if st.button("📂 Load Batch Results"):
                st.session_state.batch_id = selected_batch
                st.success(f"✅ Loaded batch {selected_batch}")
                st.rerun()
        else:
            st.info("No previous batch results found")
    
    # Research mode selection (for single requests)
    if processing_mode == "Single Request":
        research_mode = st.radio(
            "Mode:",
            ["Direct Scraping", "Research First"],
            help="Direct: Use Firecrawl immediately | Research: Use web search first"
        )
    else:
        # Initialize research_mode for batch processing (not used but prevents errors)
        research_mode = "Direct Scraping"
    
    # Initialize variables to prevent NameError
    search_provider = None
    openai_model = None
    perplexity_model = None
    use_structured_output = False
    
    if processing_mode == "Single Request" and research_mode == "Research First":
        st.subheader("🔍 Research Options")
        
        # Web search provider selection
        search_provider = st.selectbox(
            "Search Provider:",
            ["OpenAI", "Perplexity"],
            help="Choose the web search provider"
        )
        
        if search_provider == "OpenAI":
            openai_model = st.selectbox(
                "OpenAI Model:",
                ["gpt-4.1", "o4-mini"],
                help="Choose the OpenAI model for web search"
            )
        else:
            perplexity_model = st.selectbox(
                "Perplexity Model:",
                ["sonar-pro", "sonar-medium-online", "llama-3.1-sonar-small-128k-online"],
                help="Choose the Perplexity AI model for research"
            )
            
            # Structured output configuration
            st.subheader("📊 Structured Output Configuration")
            use_structured_output = st.checkbox(
                "Enable Structured Output",
                help="Get research results in a structured JSON format"
            )
            
            # Initialize session state for fields
            if 'structured_fields' not in st.session_state:
                st.session_state.structured_fields = []
            
            if use_structured_output:
                st.markdown("**Configure Data Fields:**")
                
                # Add new field button
                col_add, col_clear = st.columns([1, 1])
                with col_add:
                    if st.button("➕ Add Field", key="add_field_btn"):
                        st.session_state.structured_fields.append({
                            'name': '',
                            'type': 'string',
                            'description': '',
                            'required': False
                        })
                        st.rerun()
                
                with col_clear:
                    if st.button("🗑️ Clear All", key="clear_all_btn"):
                        st.session_state.structured_fields = []
                        st.rerun()
                
                # Display and edit existing fields
                if st.session_state.structured_fields:
                    for i, field in enumerate(st.session_state.structured_fields):
                        with st.container():
                            st.markdown(f"**Field {i+1}:**")
                            
                            # Create columns for field configuration
                            field_col1, field_col2, field_col3 = st.columns([2, 2, 1])
                            
                            with field_col1:
                                new_name = st.text_input(
                                    "Field Name:",
                                    value=field.get('name', ''),
                                    key=f"field_name_{i}",
                                    placeholder="e.g., title, summary, urls"
                                )
                                st.session_state.structured_fields[i]['name'] = new_name
                                
                                new_type = st.selectbox(
                                    "Type:",
                                    ["string", "number", "boolean", "array", "object"],
                                    index=["string", "number", "boolean", "array", "object"].index(field.get('type', 'string')),
                                    key=f"field_type_{i}"
                                )
                                st.session_state.structured_fields[i]['type'] = new_type
                            
                            with field_col2:
                                new_description = st.text_area(
                                    "Description:",
                                    value=field.get('description', ''),
                                    key=f"field_desc_{i}",
                                    height=60,
                                    placeholder="Describe what this field should contain"
                                )
                                st.session_state.structured_fields[i]['description'] = new_description
                                
                                new_required = st.checkbox(
                                    "Required",
                                    value=field.get('required', False),
                                    key=f"field_req_{i}"
                                )
                                st.session_state.structured_fields[i]['required'] = new_required
                            
                            with field_col3:
                                st.write("")  # Empty space for alignment
                                if st.button("🗑️", key=f"remove_{i}", help="Remove this field"):
                                    st.session_state.structured_fields.pop(i)
                                    st.rerun()
                            
                            st.divider()
                    
                    # Preview generated schema
                    valid_fields = [f for f in st.session_state.structured_fields if f.get('name', '').strip()]
                    if valid_fields:
                        with st.expander("📋 Preview Schema"):
                            schema = generate_json_schema(valid_fields)
                            st.json(schema)
                    else:
                        st.info("💡 Add field names to see the schema preview")
    
    st.subheader("🌐 Scraping Options")
    
    # Format selection
    formats = st.multiselect(
        "Select output formats:",
        ["markdown", "html", "json", "screenshot"],
        default=["markdown"]
    )
    
    # Crawl vs Scrape
    action_type = st.radio(
        "Action type:",
        ["Scrape URL", "Crawl Website"],
        help="Scrape: Single page | Crawl: Multiple pages"
    )
    
    if action_type == "Crawl Website":
        crawl_limit = st.number_input(
            "Crawl limit:",
            min_value=1,
            max_value=100,
            value=10,
            help="Maximum number of pages to crawl"
        )
    
    # Advanced options
    with st.expander("Advanced Options"):
        only_main_content = st.checkbox(
            "Only main content",
            value=True,
            help="Extract only the main content, excluding navigation, footer, etc."
        )
        
        timeout = st.number_input(
            "Timeout (seconds):",
            min_value=30,
            max_value=300,
            value=60,
            help="Maximum time to wait for the page to load"
        )

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📝 Input")
    
    if processing_mode == "Batch Processing":
        st.subheader("📁 Upload CSV File")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file with city names:",
            type=['csv'],
            help="Upload a CSV file with one city name per row. The first column should contain city names."
        )
        
        if uploaded_file is not None:
            try:
                # Read CSV file
                df = pd.read_csv(uploaded_file)
                
                # Show CSV preview
                st.subheader("📋 CSV Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Column mapping configuration
                st.subheader("🔧 Column Mapping")
                st.markdown("Map CSV columns to variables you can use in your query template")
                
                # Initialize session state for column mappings
                if 'column_mappings' not in st.session_state:
                    st.session_state.column_mappings = {}
                
                # Create column mapping interface
                st.markdown("**Map columns to variables:**")
                
                # Add new mapping button
                col_add_map, col_clear_map = st.columns([1, 1])
                with col_add_map:
                    if st.button("➕ Add Column Mapping", key="add_mapping_btn"):
                        st.session_state.column_mappings[f"var_{len(st.session_state.column_mappings)}"] = {
                            'column': '',
                            'variable': ''
                        }
                        st.rerun()
                
                with col_clear_map:
                    if st.button("🗑️ Clear All Mappings", key="clear_mappings_btn"):
                        st.session_state.column_mappings = {}
                        st.rerun()
                
                # Display and edit existing mappings
                if st.session_state.column_mappings:
                    for i, (key, mapping) in enumerate(st.session_state.column_mappings.items()):
                        with st.container():
                            st.markdown(f"**Mapping {i+1}:**")
                            
                            # Create columns for mapping configuration
                            map_col1, map_col2, map_col3 = st.columns([2, 2, 1])
                            
                            with map_col1:
                                # Calculate the correct index for the selectbox
                                column_options = [''] + df.columns.tolist()
                                current_column = mapping.get('column', '')
                                if current_column in df.columns.tolist():
                                    column_index = column_options.index(current_column)
                                else:
                                    column_index = 0
                                
                                new_column = st.selectbox(
                                    "CSV Column:",
                                    column_options,
                                    index=column_index,
                                    key=f"mapping_column_{i}",
                                    help="Select the CSV column to map"
                                )
                                st.session_state.column_mappings[key]['column'] = new_column
                                
                            with map_col2:
                                new_variable = st.text_input(
                                    "Variable Name:",
                                    value=mapping.get('variable', ''),
                                    key=f"mapping_variable_{i}",
                                    placeholder="e.g., city, state, name",
                                    help="Variable name to use in query template (e.g., {city})"
                                )
                                st.session_state.column_mappings[key]['variable'] = new_variable
                            
                            with map_col3:
                                st.write("")  # Empty space for alignment
                                if st.button("🗑️", key=f"remove_mapping_{i}", help="Remove this mapping"):
                                    st.session_state.column_mappings.pop(key)
                                    st.rerun()
                            
                            st.divider()
                    
                    # Show available variables
                    valid_mappings = {m['variable']: m['column'] for m in st.session_state.column_mappings.values() 
                                    if m['column'] and m['variable']}
                    if valid_mappings:
                        st.markdown("**Available variables for your query template:**")
                        for var, col in valid_mappings.items():
                            st.markdown(f"- `{{{var}}}` → column '{col}'")
                    else:
                        st.info("💡 Add column mappings to see available variables")
                
                # Row selection configuration
                st.subheader("📊 Row Selection")
                
                # Show total rows info
                total_rows = len(df)
                st.info(f"📋 Total rows in CSV: {total_rows}")
                
                # Simple row selection with + - controls
                max_rows = st.number_input(
                    "Number of rows to process:",
                    min_value=1,
                    max_value=total_rows,
                    value=min(10, total_rows),
                    step=1,
                    help="Use + and - buttons to select how many rows to process"
                )
                
                selected_rows = df.head(max_rows)
                
                # Simple visual feedback
                st.success(f"✅ Selected {max_rows} rows for processing")
                
                # Show preview with better formatting
                with st.expander("👀 Preview Selected Rows", expanded=False):
                    # Add row numbers to the preview
                    preview_df = selected_rows.copy()
                    preview_df.insert(0, 'Row #', range(len(preview_df)))
                    
                    # Format the preview
                    st.dataframe(
                        preview_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Show selection summary
                    col_summary1, col_summary2, col_summary3 = st.columns(3)
                    with col_summary1:
                        st.metric("Total Rows", total_rows)
                    with col_summary2:
                        st.metric("Selected", len(selected_rows))
                    with col_summary3:
                        st.metric("Percentage", f"{(len(selected_rows)/total_rows)*100:.1f}%")
                
                # Research query template
                st.subheader("🔍 Research Query Template")
                st.markdown("Use the variables you mapped above as placeholders in your query")
                
                # Show available variables for template
                valid_mappings = {m['variable']: m['column'] for m in st.session_state.column_mappings.values() 
                                if m['column'] and m['variable']}
                if valid_mappings:
                    st.markdown("**Available variables:** " + ", ".join([f"`{{{var}}}`" for var in valid_mappings.keys()]))
                else:
                    st.warning("⚠️ No column mappings configured. Add mappings above to use variables in your query.")
                
                research_query_template = st.text_area(
                    "Research query template:",
                    value="Research information about {city} in {state}.",
                    height=100,
                    help="Use the variables you mapped above as placeholders (e.g., {city}, {state})"
                )
                
                # Show query preview for first few rows
                if research_query_template and len(selected_rows) > 0 and valid_mappings:
                    with st.expander("👀 Query Preview"):
                        st.markdown("**How the query will look for the first 3 rows:**")
                        for i, (idx, row) in enumerate(selected_rows.head(3).iterrows()):
                            preview_query = research_query_template
                            # Replace all variable placeholders with actual values
                            for var, col in valid_mappings.items():
                                placeholder = f"{{{var}}}"
                                if placeholder in preview_query:
                                    value = str(row[col]) if not pd.isna(row[col]) else ""
                                    preview_query = preview_query.replace(placeholder, value)
                            
                            st.markdown(f"**Row {i+1} (index {idx}):**")
                            st.text(preview_query)
                            st.divider()
                
                # Structured output configuration for batch
                st.subheader("📊 Structured Output Configuration")
                use_batch_structured_output = st.checkbox(
                    "Enable Structured Output",
                    help="Get research results in a structured JSON format"
                )
                
                # Initialize session state for batch fields
                if 'batch_structured_fields' not in st.session_state:
                    st.session_state.batch_structured_fields = []
                
                if use_batch_structured_output:
                    st.markdown("**Configure Data Fields:**")
                    
                    # Add new field button
                    col_add, col_clear = st.columns([1, 1])
                    with col_add:
                        if st.button("➕ Add Field", key="add_batch_field_btn"):
                            st.session_state.batch_structured_fields.append({
                                'name': '',
                                'type': 'string',
                                'description': '',
                                'required': False
                            })
                            st.rerun()
                    
                    with col_clear:
                        if st.button("🗑️ Clear All", key="clear_batch_all_btn"):
                            st.session_state.batch_structured_fields = []
                            st.rerun()
                    
                    # Display and edit existing fields
                    if st.session_state.batch_structured_fields:
                        for i, field in enumerate(st.session_state.batch_structured_fields):
                            with st.container():
                                st.markdown(f"**Field {i+1}:**")
                                
                                # Create columns for field configuration
                                field_col1, field_col2, field_col3 = st.columns([2, 2, 1])
                                
                                with field_col1:
                                    new_name = st.text_input(
                                        "Field Name:",
                                        value=field.get('name', ''),
                                        key=f"batch_field_name_{i}",
                                        placeholder="e.g., clerk_name, title, contact"
                                    )
                                    st.session_state.batch_structured_fields[i]['name'] = new_name
                                    
                                    new_type = st.selectbox(
                                        "Type:",
                                        ["string", "number", "boolean", "array", "object"],
                                        index=["string", "number", "boolean", "array", "object"].index(field.get('type', 'string')),
                                        key=f"batch_field_type_{i}"
                                    )
                                    st.session_state.batch_structured_fields[i]['type'] = new_type
                                
                                with field_col2:
                                    new_description = st.text_area(
                                        "Description:",
                                        value=field.get('description', ''),
                                        key=f"batch_field_desc_{i}",
                                        height=60,
                                        placeholder="Describe what this field should contain"
                                    )
                                    st.session_state.batch_structured_fields[i]['description'] = new_description
                                    
                                    new_required = st.checkbox(
                                        "Required",
                                        value=field.get('required', False),
                                        key=f"batch_field_req_{i}"
                                    )
                                    st.session_state.batch_structured_fields[i]['required'] = new_required
                                
                                with field_col3:
                                    st.write("")  # Empty space for alignment
                                    if st.button("🗑️", key=f"batch_remove_{i}", help="Remove this field"):
                                        st.session_state.batch_structured_fields.pop(i)
                                        st.rerun()
                                
                                st.divider()
                    
                    # Preview generated schema
                    valid_fields = [f for f in st.session_state.batch_structured_fields if f.get('name', '').strip()]
                    if valid_fields:
                        with st.expander("📋 Preview Schema"):
                            schema = generate_json_schema(valid_fields)
                            st.json(schema)
                    else:
                        st.info("💡 Add field names to see the schema preview")
                
                # Start batch processing button
                if st.button("🚀 Start Batch Processing", type="primary", use_container_width=True):
                    if not research_query_template:
                        st.error("Please enter a research query template")
                    else:
                        # Prepare structured fields if enabled
                        structured_fields = None
                        if use_batch_structured_output:
                            valid_fields = [f for f in st.session_state.get('batch_structured_fields', []) if f.get('name', '').strip()]
                            if valid_fields:
                                structured_fields = valid_fields
                            elif use_batch_structured_output:
                                st.warning("⚠️ No valid fields configured for structured output")
                        
                        # Get the correct model based on provider
                        model = batch_openai_model if batch_search_provider == "OpenAI" else batch_perplexity_model
                        
                        # Prepare column mappings
                        valid_mappings = {m['variable']: m['column'] for m in st.session_state.column_mappings.values() 
                                        if m['column'] and m['variable']}
                        
                        with st.spinner(f"Starting batch processing for {len(selected_rows)} rows..."):
                            batch_id, results = process_batch_cities(
                                selected_rows.index.tolist(), # Pass row indices
                                research_query_template, 
                                batch_search_provider, 
                                model, 
                                structured_fields, 
                                request_delay,
                                selected_rows, # Pass selected_rows
                                valid_mappings # Pass valid column mappings
                            )
                            
                            st.session_state.batch_id = batch_id
                            st.session_state.batch_results = results
                            st.success(f"✅ Batch processing completed! Batch ID: {batch_id}")
                
                # Show output folder info
                st.info(f"📁 Results will be saved to: `{OUTPUT_DIR.absolute()}/[batch_id]/`")
                
            except Exception as e:
                st.error(f"❌ Error reading CSV file: {str(e)}")
    
    elif processing_mode == "Single Request":
        if research_mode == "Research First":
            # Research query input
            research_query = st.text_area(
                "Enter your research query:",
                placeholder="Research the latest AI developments and find relevant URLs to analyze...",
                height=100,
                help="Describe what you want to research. The selected search provider will help find relevant information and URLs."
            )
        
        # Structured output configuration
        st.subheader("📊 Structured Output Configuration")
        use_structured_output = st.checkbox(
            "Enable Structured Output",
            help="Get research results in a structured JSON format"
        )
        
        # Initialize session state for fields
        if 'structured_fields' not in st.session_state:
            st.session_state.structured_fields = []
        
        if use_structured_output:
            st.markdown("**Configure Data Fields:**")
            
            # Add new field button
            col_add, col_clear = st.columns([1, 1])
            with col_add:
                if st.button("➕ Add Field", key="add_field_btn"):
                    st.session_state.structured_fields.append({
                        'name': '',
                        'type': 'string',
                        'description': '',
                        'required': False
                    })
                    st.rerun()
            
            with col_clear:
                if st.button("🗑️ Clear All", key="clear_all_btn"):
                    st.session_state.structured_fields = []
                    st.rerun()
            
            # Display and edit existing fields
            if st.session_state.structured_fields:
                for i, field in enumerate(st.session_state.structured_fields):
                    with st.container():
                        st.markdown(f"**Field {i+1}:**")
                        
                        # Create columns for field configuration
                        field_col1, field_col2, field_col3 = st.columns([2, 2, 1])
                        
                        with field_col1:
                            new_name = st.text_input(
                                "Field Name:",
                                value=field.get('name', ''),
                                key=f"field_name_{i}",
                                placeholder="e.g., title, summary, urls"
                            )
                            st.session_state.structured_fields[i]['name'] = new_name
                            
                            new_type = st.selectbox(
                                "Type:",
                                ["string", "number", "boolean", "array", "object"],
                                index=["string", "number", "boolean", "array", "object"].index(field.get('type', 'string')),
                                key=f"field_type_{i}"
                            )
                            st.session_state.structured_fields[i]['type'] = new_type
                        
                        with field_col2:
                            new_description = st.text_area(
                                "Description:",
                                value=field.get('description', ''),
                                key=f"field_desc_{i}",
                                height=60,
                                placeholder="Describe what this field should contain"
                            )
                            st.session_state.structured_fields[i]['description'] = new_description
                            
                            new_required = st.checkbox(
                                "Required",
                                value=field.get('required', False),
                                key=f"field_req_{i}"
                            )
                            st.session_state.structured_fields[i]['required'] = new_required
                        
                        with field_col3:
                            st.write("")  # Empty space for alignment
                            if st.button("🗑️", key=f"remove_{i}", help="Remove this field"):
                                st.session_state.structured_fields.pop(i)
                                st.rerun()
                        
                        st.divider()
                
                # Preview generated schema
                valid_fields = [f for f in st.session_state.structured_fields if f.get('name', '').strip()]
                if valid_fields:
                    with st.expander("📋 Preview Schema"):
                        schema = generate_json_schema(valid_fields)
                        st.json(schema)
                else:
                    st.info("💡 Add field names to see the schema preview")
        
        # Research button
        if st.button("🔍 Start Research", type="primary", use_container_width=True):
            if not research_query:
                st.error("Please enter a research query")
            else:
                # Prepare structured fields if enabled
                structured_fields = None
                if use_structured_output:
                    valid_fields = [f for f in st.session_state.get('structured_fields', []) if f.get('name', '').strip()]
                    if valid_fields:
                        structured_fields = valid_fields
                    elif use_structured_output:
                        st.warning("⚠️ No valid fields configured for structured output")
                
                with st.spinner(f"Researching with {search_provider}..."):
                    if search_provider == "OpenAI":
                        research_result = research_with_openai(research_query, openai_model, structured_fields)
                    else:
                        research_result = research_with_perplexity(research_query, perplexity_model, structured_fields)
                    
                    if research_result:
                        st.session_state.research_result = research_result
                        st.success("✅ Research completed!")
                        
                        # Extract URLs from research for easy selection
                        content = research_result['content']
                        st.session_state.research_content = content
                        
                        # Try to extract URLs from the response
                        import re
                        urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', content)
                        if urls:
                            st.session_state.found_urls = urls
                            st.info(f"🔗 Found {len(urls)} URLs in research results")
    
    # URL input
    url = st.text_input(
        "Enter URL to scrape:",
        placeholder="https://example.com",
        help="Enter the full URL including http:// or https://"
    )
    
    # Show found URLs from research if available
    if hasattr(st.session_state, 'found_urls') and st.session_state.found_urls:
        st.subheader("🔗 URLs Found in Research")
        for i, found_url in enumerate(st.session_state.found_urls[:5]):  # Show first 5 URLs
            if st.button(f"Use: {found_url[:50]}...", key=f"url_{i}"):
                st.session_state.selected_url = found_url
                st.rerun()
        
        if st.session_state.get('selected_url'):
            url = st.session_state.selected_url
            st.success(f"✅ Selected URL: {url}")
    
    # Scrape button
    button_text = "🚀 Start Scraping" if research_mode == "Direct Scraping" else "🌐 Scrape Selected URL"
    if st.button(button_text, type="primary", use_container_width=True):
        if not url:
            st.error("Please enter a URL")
        elif not formats:
            st.error("Please select at least one output format")
        else:
            with st.spinner("Scraping in progress..."):
                try:
                    app = get_firecrawl_app()
                    if app is None:
                        st.stop()
                    
                    if action_type == "Scrape URL":
                        # Single page scraping
                        result = app.scrape_url(
                            url,
                            formats=formats,
                            only_main_content=only_main_content,
                            timeout=timeout * 1000  # Convert to milliseconds
                        )
                        
                        if result.success:
                            st.success("✅ Scraping completed successfully!")
                        else:
                            st.error("❌ Scraping failed")
                            
                    else:
                        # Website crawling
                        crawl_result = app.crawl_url(
                            url,
                            limit=crawl_limit,
                            scrape_options=ScrapeOptions(
                                formats=formats,
                                only_main_content=only_main_content,
                                timeout=timeout * 1000
                            )
                        )
                        
                        if crawl_result.success:
                            st.success(f"✅ Crawl job started! Job ID: {crawl_result.id}")
                            
                            # Check crawl status
                            with st.spinner("Checking crawl status..."):
                                status_result = app.check_crawl_status(crawl_result.id)
                                
                                if status_result.status == "completed":
                                    st.success("✅ Crawling completed!")
                                    result = status_result
                                else:
                                    st.info(f"⏳ Crawl in progress: {status_result.completed}/{status_result.total} pages completed")
                                    result = status_result
                        else:
                            st.error("❌ Crawl failed")
                    
                    # Store result in session state
                    st.session_state.scrape_result = result
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

with col2:
    st.header("📊 Results")
    
    # Display batch results if available
    if processing_mode == "Batch Processing" and hasattr(st.session_state, 'batch_id') and st.session_state.batch_id:
        with st.expander("📊 Batch Processing Results", expanded=True):
            batch_id = st.session_state.batch_id
            
            # Load results from files
            batch_results = load_batch_results(batch_id)
            
            if batch_results:
                st.success(f"✅ Batch {batch_id} completed with {len(batch_results)} results")
                
                # Show batch folder info
                batch_folder = OUTPUT_DIR / batch_id
                st.info(f"📁 Results saved in: `{batch_folder.absolute()}`")
                
                # Create summary table
                summary_df = create_batch_summary(batch_results)
                st.markdown("### 📋 Summary Table")
                st.dataframe(summary_df, use_container_width=True)
                
                # Download options
                col_download1, col_download2 = st.columns(2)
                with col_download1:
                    # Download as CSV
                    csv_data = summary_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name=f"batch_results_{batch_id}.csv",
                        mime="text/csv"
                    )
                
                with col_download2:
                    # Download as JSON
                    json_data = json.dumps(batch_results, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_data,
                        file_name=f"batch_results_{batch_id}.json",
                        mime="application/json"
                    )
                
                # Show individual results
                st.markdown("### 🔍 Individual Results")
                for result in batch_results:
                    row_name = result.get('row_name', 'Unknown')
                    row_index = result.get('row_index', 'Unknown')
                    
                    # Use a container instead of expander to avoid nesting
                    with st.container():
                        st.markdown(f"#### 📄 Row {row_index}: {row_name}")
                        result_data = result.get('result', {})
                        
                        # Show the actual query that was sent
                        if 'query' in result:
                            st.markdown("**🔍 Query Sent:**")
                            st.text(result['query'])
                            st.divider()
                        
                        if result_data.get('structured_data'):
                            st.markdown("**📊 Structured Data:**")
                            st.json(result_data['structured_data'])
                        
                        st.markdown("**📝 Raw Response:**")
                        st.markdown(result_data.get('content', 'No content available'))
                        
                        # Show provider info
                        provider = result_data.get('provider', 'unknown')
                        st.markdown(f"**Provider:** {provider.title()}")
                        
                        st.divider()  # Add separator between results
            else:
                st.warning("⚠️ No batch results found. Check the output folder for saved files.")
    
    # Display research results if available
    if hasattr(st.session_state, 'research_result') and st.session_state.research_result:
        with st.expander("🔍 Research Results", expanded=True):
            research_result = st.session_state.research_result
            provider = research_result.get('provider', 'unknown')
            
            # Display research content
            st.markdown(f"### Research Response ({provider.title()})")
            
            # Show structured data if available
            if research_result.get('structured_data'):
                st.markdown("#### 📊 Structured Data")
                st.json(research_result['structured_data'])
                
                # Create tabs for structured vs raw content
                tab1, tab2 = st.tabs(["Structured Data", "Raw Response"])
                
                with tab1:
                    st.json(research_result['structured_data'])
                
                with tab2:
                    st.markdown(research_result['content'])
            else:
                st.markdown(research_result['content'])
            
            # Show usage info if available (mainly for Perplexity)
            if provider == "perplexity" and 'raw_response' in research_result:
                raw_response = research_result['raw_response']
                if 'usage' in raw_response:
                    usage = raw_response['usage']
                    st.markdown("### API Usage")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Prompt Tokens", usage.get('prompt_tokens', 0))
                    with col2:
                        st.metric("Completion Tokens", usage.get('completion_tokens', 0))
                    with col3:
                        st.metric("Total Tokens", usage.get('total_tokens', 0))
    
    # Display scraping results
    if hasattr(st.session_state, 'scrape_result') and st.session_state.scrape_result:
        result = st.session_state.scrape_result
        
        # Create tabs for different formats
        if hasattr(result, 'data') and result.data:
            data = result.data
            
            # Create tabs for different output formats
            if formats:
                tab_names = [f.capitalize() for f in formats if f in data]
                if tab_names:
                    tabs = st.tabs(tab_names)
                    
                    for i, format_name in enumerate(tab_names):
                        format_key = format_name.lower()
                        with tabs[i]:
                            if format_key == "markdown" and "markdown" in data:
                                st.markdown("### Markdown Content")
                                st.text_area(
                                    "Markdown:",
                                    value=data["markdown"],
                                    height=400,
                                    key=f"markdown_{i}"
                                )
                                
                            elif format_key == "html" and "html" in data:
                                st.markdown("### HTML Content")
                                st.text_area(
                                    "HTML:",
                                    value=data["html"],
                                    height=400,
                                    key=f"html_{i}"
                                )
                                
                            elif format_key == "json" and "json" in data:
                                st.markdown("### JSON Content")
                                st.json(data["json"])
                                
                            elif format_key == "screenshot" and "screenshot" in data:
                                st.markdown("### Screenshot")
                                st.image(data["screenshot"], caption="Page Screenshot")
            
            # Display metadata
            if "metadata" in data:
                with st.expander("📋 Metadata"):
                    st.json(data["metadata"])
        
        # For crawl results with multiple pages
        elif hasattr(result, 'data') and isinstance(result.data, list):
            st.markdown(f"### Crawl Results ({len(result.data)} pages)")
            
            for i, page_data in enumerate(result.data):
                with st.expander(f"Page {i+1}: {page_data.get('metadata', {}).get('title', 'Untitled')}"):
                    if "markdown" in page_data:
                        st.markdown("**Markdown:**")
                        st.text_area(
                            f"Markdown {i+1}:",
                            value=page_data["markdown"][:500] + "..." if len(page_data["markdown"]) > 500 else page_data["markdown"],
                            height=200,
                            key=f"crawl_markdown_{i}"
                        )
                    
                    if "metadata" in page_data:
                        st.markdown("**Metadata:**")
                        st.json(page_data["metadata"])
        
        # Display raw result for debugging
        with st.expander("🔍 Raw Response"):
            st.json(result.__dict__ if hasattr(result, '__dict__') else result)

# Footer
st.markdown("---")
st.markdown(
    "Built with ❤️ using [Streamlit](https://streamlit.io) and [Firecrawl](https://firecrawl.dev)"
) 