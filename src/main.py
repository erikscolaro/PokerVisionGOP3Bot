import ImageAnalyzer
import os
import cv2
import json 

query_path = os.path.join("assets", "query", "query.json")

def main():
    # Define path to tesseract executable
    tesseract_path = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

    # Initialize the analyzer with the tesseract path
    analyzer = ImageAnalyzer.Recognizer(tesseract_path)

    try:
        # Process the query and get the results
        results = analyzer.process_query(job_path=query_path)
        
        # Handle results
        if results:
            print(f"Found {len(results)} matches")
            print(json.dumps(results, indent=2))
        else:
            print("No matches found or processing completed with no results")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
