import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def fetch_html(url):
    response = requests.get(url)
    return response.text

def parse_html(html_content, base_url):
    soup = BeautifulSoup(html_content, 'html.parser')
    content = []
    
    # Extract text and code
    for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'code']):
        if element.name == 'code':
            content.append(('code', element.get_text()))
        else:
            content.append(('text', element.get_text()))
    
    # Extract links for sub-paths
    links = soup.find_all('a', href=True)
    sub_paths = []
    for link in links:
        full_url = urljoin(base_url, link['href'])
        if full_url.startswith(base_url) and full_url != base_url:
            sub_paths.append(full_url)
    
    return content, sub_paths

def convert_to_markdown(content):
    markdown_content = ""
    for content_type, text in content:
        if content_type == 'text':
            markdown_content += f"{text}\n\n"
        elif content_type == 'code':
            markdown_content += f"```\n{text}\n```\n\n"
    return markdown_content

def save_to_markdown(content, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

def process_url(url, base_url, visited_urls, output_file):
    if url in visited_urls:
        return

    visited_urls.add(url)
    print(f"Processing: {url}")

    html_content = fetch_html(url)
    parsed_content, sub_paths = parse_html(html_content, base_url)
    markdown_content = convert_to_markdown(parsed_content)
    
    # Append to the output file
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"# {url}\n\n")
        f.write(markdown_content)
        f.write("\n---\n\n")

    # Process sub-paths
    for sub_path in sub_paths:
        process_url(sub_path, base_url, visited_urls, output_file)

def main():
    base_url = input("Enter the base URL to fetch: ")
    output_file = input("Enter the output Markdown filename: ")

    # Ensure the output file is empty
    open(output_file, 'w').close()

    visited_urls = set()
    process_url(base_url, base_url, visited_urls, output_file)

    print(f"Markdown file '{output_file}' has been created with content from {len(visited_urls)} pages.")

if __name__ == "__main__":
    main()