import requests
import json

url = "http://localhost:8000/analyze-project"
file_path = "Restaurantsolution.zip"
# file_path = "Restaurantsolution2.zip"
# file_path = "destinationsolution.zip"
# file_path = "contactbook.zip"


project_about = "Restaurant Application"
technology = "HTML/CSS/JavaScript"
problem_statement = """Create a responsive Restaurant web application with a Bootstrap-integrated navbar for navigation, search functionality, and filter buttons for categorizing and searching menu items."""

scoring_pattern = [
    {"component": "Navbar", "max_score": 20},
    {"component": "Search Functionality", "max_score": 30},
    {"component": "Responsive Design", "max_score": 30},
    {"component": "Filter buttons for categorizing.", "max_score": 20}
]

with open(file_path, "rb") as f:
    files = {"zip_file": (file_path, f)}
    data = {
        "project_about": project_about,
        "technology": technology,
        "problem_statement": problem_statement,
        "scoring_pattern": json.dumps(scoring_pattern)
    }
    response = requests.post(url, files=files, data=data)

print(response.json())