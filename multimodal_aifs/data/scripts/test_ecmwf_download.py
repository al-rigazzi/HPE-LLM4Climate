"""
Simple ECMWF Data Download Test

Test downloading real weather data from ECMWF Open Data with a simpler approach.
"""

from datetime import datetime, timedelta

import requests
from ecmwf.opendata import Client


def test_simple_download():
    """Test a simple download from ECMWF Open Data."""
    print("🌍 Testing ECMWF Open Data Download")
    print("=" * 50)

    try:
        client = Client()

        # Get yesterday's date
        yesterday = datetime.now() - timedelta(days=1)
        date_str = yesterday.strftime("%Y-%m-%d")

        print(f"📅 Downloading data for: {date_str}")
        print("📊 Variables: 2m temperature")
        print("🌍 Area: Global")
        print("⏰ Time: 12Z")

        # Simple download request
        filename = f"test_download_{date_str}.grib"

        client.retrieve(
            type="fc",
            step=6,
            param="2t",  # 2m temperature only
            target=filename,
            date=date_str,
            time=12,  # Use integer instead of string
        )

        print(f"✅ Successfully downloaded: {filename}")

        # Check file size
        import os

        if os.path.exists(filename):
            size_mb = os.path.getsize(filename) / (1024**2)
            print(f"📁 File size: {size_mb:.1f} MB")

    except Exception as e:
        print(f"❌ Download failed: {e}")
        print(f"   Error type: {type(e).__name__}")

        # Try to get available data info
        try:
            print("\n🔍 Checking available data...")
            # This is a simple test URL
            test_url = "https://data.ecmwf.int/forecasts/"
            response = requests.get(test_url, timeout=5)
            print(f"   ECMWF service status: {response.status_code}")
        except Exception as e2:
            print(f"   Could not check service: {e2}")


def test_data_availability():
    """Test what data is available."""
    print("\n📋 Testing Data Availability")
    print("=" * 30)

    try:
        client = Client()

        # Get recent dates
        dates = []
        for i in range(1, 8):  # Last week
            date = datetime.now() - timedelta(days=i)
            dates.append(date.strftime("%Y-%m-%d"))

        print("🗓️  Testing availability for recent dates:")
        for date_str in dates[:3]:  # Test first 3 dates
            try:
                print(f"   {date_str}: Testing...")
                # Try a very simple request
                client.retrieve(
                    type="fc",
                    step=0,
                    param="2t",
                    date=date_str,
                    time=0,
                    target=f"test_{date_str}.grib",
                )
                print(f"   {date_str}: ✅ Available")
                break
            except Exception as e:
                print(f"   {date_str}: ❌ {str(e)[:50]}...")

    except Exception as e:
        print(f"❌ Availability check failed: {e}")


if __name__ == "__main__":
    test_simple_download()
    test_data_availability()
