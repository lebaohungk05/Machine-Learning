import os
import requests
import base64
import json

def generate_mermaid_image(mermaid_code, output_file):
    # Encode the Mermaid code
    graphbytes = mermaid_code.encode("utf8")
    base64_graph = base64.b64encode(graphbytes).decode("utf8")
    
    # Create the URL for the Mermaid Live Editor
    url = f"https://mermaid.ink/img/{base64_graph}?type=png"
    
    # Download the image
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"Image saved to {output_file}")
    else:
        print("Failed to generate image")

# Read the Mermaid code from the markdown file
with open("system_diagram.md", "r", encoding="utf-8") as f:
    content = f.read()
    
# Extract the Mermaid code (between ```mermaid and ```)
mermaid_code = content.split("```mermaid")[1].split("```")[0].strip()

# Generate the image
generate_mermaid_image(mermaid_code, "system_diagram.png") 