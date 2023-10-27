import requests
import csv
import time
import os

base_url = "https://cs7ns1.scss.tcd.ie/?shortname=alammu&myfilename="

def download_image_with_retry(filename, max_retries=5):
    retry_delay = 1  # Initial retry delay in seconds

    for retry_count in range(max_retries):
        url = base_url + filename
        response = requests.get(url)

        if response.status_code == 200:
            # Image successfully downloaded
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {filename}")
            return True

        # Handle 403 Forbidden status code (retry)
        elif response.status_code == 403:
            print(f"Retry {retry_count + 1}/{max_retries} for: {filename}")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
        else:
            print(f"Failed to download: {filename}")
            return False

    return False

csv_filename = "filenames.csv"

if not os.path.exists(csv_filename):
    print(f"CSV file '{csv_filename}' not found.")
    exit(1)

failed_filenames = []

with open(csv_filename, 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip header if present
    for row in csv_reader:
        if len(row) > 0:
            filename = row[0].strip()
            if not download_image_with_retry(filename):
                failed_filenames.append(filename)

print("Download process completed.")

if failed_filenames:
    failed_csv_filename = "failed_filenames.csv"
    with open(failed_csv_filename, 'w', newline='') as failed_file:
        csv_writer = csv.writer(failed_file)
        csv_writer.writerow(["Failed Filenames"])
        csv_writer.writerows([[filename] for filename in failed_filenames])

    print(f"Failed filenames saved to '{failed_csv_filename}'")
else:
    print("No files failed to download.")

