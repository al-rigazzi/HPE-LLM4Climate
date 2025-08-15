# Demo Tests

This directory contains demonstration scripts that showcase various capabilities of the climate analysis system.

## 🎯 Demo Files

### Location-Aware Demonstrations
- **`working_location_demo.py`** - Real-world geographic location processing example

## 🚀 Running Demos

```bash
# Run location-aware demonstration
python tests/demos/working_location_demo.py
```

## 📍 What the Demo Shows

### Geographic Processing
- 🌍 **Address Resolution**: Convert text addresses to coordinates
- 📍 **Coordinate Validation**: Verify latitude/longitude pairs
- 🗺️ **Reverse Geocoding**: Get location names from coordinates
- 🎯 **Spatial Masking**: Apply geographic filters to climate data

### Real-World Examples
The demo processes various location types:
- **Cities**: "Paris, France" → (48.8566, 2.3522)
- **Landmarks**: "Eiffel Tower" → (48.8584, 2.2945)
- **Addresses**: "1600 Pennsylvania Avenue, Washington DC" → (38.8977, -77.0365)
- **Coordinates**: "(40.7128, -74.0060)" → "New York, NY"

### Integration Showcase
- ✅ **GeoPy Integration**: Real geocoding service usage
- ✅ **Nominatim Provider**: OpenStreetMap-based resolution
- ✅ **Error Handling**: Graceful fallbacks for invalid locations
- ✅ **Location Caching**: Efficient repeated lookups
- ✅ **Multimodal Fusion**: Geographic data + climate analysis

## 🛠️ Technical Details

### Dependencies
- **GeoPy**: Geographic processing library
- **Nominatim**: OpenStreetMap geocoding service
- **Multimodal Core**: Location-aware fusion components

### Network Requirements
The demo requires internet connectivity for:
- 🌐 **Geocoding Services**: Converting addresses to coordinates
- 📡 **Reverse Geocoding**: Getting place names from coordinates
- 🗺️ **OpenStreetMap API**: Accessing Nominatim geocoding service

### Error Handling
- ⚠️ **Network Issues**: Graceful degradation when services unavailable
- 🚫 **Invalid Locations**: Clear error messages for unresolvable addresses
- 🔄 **Retry Logic**: Automatic retry for temporary failures
- 💾 **Fallback Mode**: Use mock coordinates when geocoding fails

## 📊 Expected Output

The demo shows:

```
🌍 Geographic Resolution Demo
=============================

✅ Resolving: "Paris, France"
   → Coordinates: (48.8566, 2.3522)
   → Country: France

✅ Resolving: "Eiffel Tower"
   → Coordinates: (48.8584, 2.2945)
   → Address: Tour Eiffel, Paris, France

✅ Processing coordinates: (40.7128, -74.0060)
   → Location: New York, NY, USA
   → Time zone: America/New_York

📍 All location processing successful!
```

## 🔧 Troubleshooting

### Common Issues
1. **Network Connectivity**: Ensure internet access for geocoding
2. **Rate Limiting**: Nominatim has usage limits; add delays if needed
3. **Invalid Locations**: Some addresses may not be found in OpenStreetMap

### Alternative Usage
```bash
# Run with custom locations
python tests/demos/working_location_demo.py --location "Your Address Here"

# Run in offline mode (uses mock coordinates)
python tests/demos/working_location_demo.py --offline
```

## 🎓 Learning Outcomes

After running this demo, you'll understand:
- 🗺️ How geographic resolution works in the climate system
- 🔗 Integration patterns between location services and climate data
- 🛡️ Error handling strategies for external service dependencies
- 📍 Best practices for location-aware climate analysis
