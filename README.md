# 🌐 Firecrawl Web Scraper

A powerful Streamlit application for web scraping and research using the Firecrawl API, with support for both single requests and batch processing.

## Features

### Single Request Mode
- **Direct Scraping**: Scrape individual URLs with Firecrawl
- **Research First**: Use web search (OpenAI/Perplexity) before scraping
- **Multiple Output Formats**: Markdown, HTML, JSON, Screenshots
- **Structured Output**: Configure custom JSON schemas for research results
- **Website Crawling**: Crawl entire websites with configurable limits

### Batch Processing Mode
- **CSV Upload**: Upload CSV files with any type of data
- **Flexible Column Mapping**: Map any CSV column to any variable name
- **Flexible Row Selection**: Choose first N rows, specific range, or all rows
- **Dynamic Queries**: Use custom variables like `{name}`, `{company}`, `{position}` in query templates
- **Progress Tracking**: Real-time progress bar and status updates
- **Automatic Saving**: Results saved to local `output/` folder
- **Resume Capability**: Load and continue previous batch sessions
- **Export Options**: Download results as CSV or JSON

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   FIRECRAWL_API_KEY=your_firecrawl_key
   OPENAI_API_KEY=your_openai_key
   PERPLEXITY_API_KEY=your_perplexity_key
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

### Single Request Mode
1. Select "Single Request" mode
2. Choose between "Direct Scraping" or "Research First"
3. Configure scraping options (formats, content type, timeout)
4. Enter URL or research query
5. Click "Start Scraping" or "Start Research"

### Batch Processing Mode
1. Select "Batch Processing" mode
2. Upload a CSV file with your data
3. **Map columns to variables**: Create custom mappings (e.g., "Name" column → "name" variable)
4. **Select rows**: Choose how many rows to process (first N, specific range, or all)
5. Create a query template using your mapped variables
6. Optionally configure structured output fields
7. Click "Start Batch Processing"

### CSV Format
Your CSV file can contain any type of data. For example:
```csv
Name,Company,Position,Location,Salary,Website,Industry
John Smith,Apple,Software Engineer,Cupertino,120000,https://apple.com,Technology
Sarah Johnson,Google,Product Manager,Mountain View,140000,https://google.com,Technology
```

### Column Mapping
Map CSV columns to variables you can use in your query:
- **CSV Column**: "Name" → **Variable**: "name" → Use as `{name}` in query
- **CSV Column**: "Company" → **Variable**: "company" → Use as `{company}` in query
- **CSV Column**: "Position" → **Variable**: "role" → Use as `{role}` in query

### Query Templates
Use your mapped variables in query templates:
- `{name}` - Replaced with the person's name
- `{company}` - Replaced with the company name
- `{position}` - Replaced with the job position
- Any other mapped variable in `{variable_name}` format

Example template:
```
Research {name} who works as {position} at {company}. Find their professional background and recent activities.
```

## Output

### Single Requests
- Results displayed in the app interface
- Multiple format tabs (Markdown, HTML, JSON, Screenshots)
- Raw response data available

### Batch Processing
- Results automatically saved to `output/` folder
- Individual JSON files for each processed item
- Summary table with all results
- Download options for CSV and JSON exports
- Query preview showing actual queries sent

## File Structure
```
├── app.py                 # Main application
├── requirements.txt       # Python dependencies
├── sample_cities.csv      # Example CSV file
├── output/               # Batch processing results
└── README.md            # This file
```

## API Keys Required
- **Firecrawl API Key**: For web scraping functionality
- **OpenAI API Key**: For OpenAI web search (optional)
- **Perplexity API Key**: For Perplexity AI research (optional)

## Notes
- Batch processing includes delays between requests to avoid rate limiting
- Results are automatically saved to prevent data loss
- Previous batch sessions can be loaded from the sidebar
- The app creates an `output/` directory automatically 