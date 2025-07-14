import requests
import json

def test_api():
    """Test the API endpoints"""
    base_url = "https://stock-be-j9p2.onrender.com"
    
    print("Testing API endpoints...")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"Health check: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
    
    # Test home endpoint
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        print(f"Home endpoint: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Home endpoint failed: {e}")
    
    # Test analyze endpoint (this will take a while)
    try:
        print("Testing analyze endpoint (this may take 3-5 minutes)...")
        response = requests.get(f"{base_url}/analyze", timeout=300)
        print(f"Analyze endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Stocks analyzed: {data.get('stocks_analyzed', 0)}")
            print(f"Status: {data.get('status', 'unknown')}")
        else:
            print(f"Error response: {response.text}")
    except Exception as e:
        print(f"Analyze endpoint failed: {e}")

if __name__ == "__main__":
    test_api()
