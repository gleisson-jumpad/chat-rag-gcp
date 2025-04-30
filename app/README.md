# Streamlit Pages Directory

This directory contains additional pages for the Streamlit multi-page application.

## How Multi-page Apps Work in Streamlit

Streamlit automatically detects Python files in the `pages/` directory and adds them as separate pages in the sidebar navigation.

- The main file (in the parent directory) becomes the home page
- Each Python file in this directory becomes a separate page
- Pages are listed in alphabetical order by filename
- You can add numbers to filenames (like `01_page.py`) to control the ordering

## Available Pages

- `direct_test.py` - Test direct connections to PostgreSQL using public IP or Unix socket methods

## How to Use

1. Run the main application
2. Use the sidebar to navigate between pages
3. Each page can maintain its own state and has its own URL 