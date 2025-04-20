import numpy as np
from sklearn.cluster import DBSCAN
import requests
import logging
import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS


load_dotenv()

app = Flask(__name__)
CORS(app)  
logging.basicConfig(level=logging.DEBUG)

# API Keys
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "APIkey") #insert api key here
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "APIkey")


FUEL_PRICE_PER_LITER = 100  # Example price in your currency
AVG_FUEL_CONSUMPTION = 10  # km/L - average consumption


@app.route('/')
def index():
    return render_template('index.html', google_maps_api_key=GOOGLE_MAPS_API_KEY)


def geocode(location):
    """Convert address to lat/lng using Google Geocoding API"""
    url = f'https://maps.googleapis.com/maps/api/geocode/json'
    params = {'address': location, 'key': GOOGLE_MAPS_API_KEY}

    try:
        response = requests.get(url, params=params)
        logging.debug(f"Geocode Response: {response.text}")
        response.raise_for_status()
        data = response.json()

        if data['status'] == 'OK' and data['results']:
            result = data['results'][0]
            location = result['geometry']['location']
            return {
                'lat': location['lat'],
                'lng': location['lng'],
                'address': result['formatted_address']
            }
        logging.warning(f"No geocode results for {location}")
    except Exception as e:
        logging.error(f"Geocoding error: {e}")
    return None


def get_weather(lat, lng):
    """Get weather information for a location"""
    url = f"http://api.weatherapi.com/v1/forecast.json"
    params = {
        'key': WEATHER_API_KEY,
        'q': f"{lat},{lng}",
        'days': 1
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        weather_data = response.json()
        # Extract only the current weather information
        current_weather = weather_data.get('current', {})
        return {
            'condition': current_weather.get('condition', {}).get('text', 'N/A'),
            'temperature_c': current_weather.get('temp_c', 'N/A'),
            'temperature_f': current_weather.get('temp_f', 'N/A'),
            'wind_kph': current_weather.get('wind_kph', 'N/A'),
            'humidity': current_weather.get('humidity', 'N/A'),
        }
    except Exception as e:
        logging.error(f"Weather API error: {e}")
        return {
            'condition': 'Unknown',
            'temperature_c': 'N/A',
            'temperature_f': 'N/A'
        }


def calculate_fuel_cost(distance_km):
    """Calculate fuel consumption and cost based on distance"""
    fuel_consumed = distance_km / AVG_FUEL_CONSUMPTION  # in liters
    return fuel_consumed * FUEL_PRICE_PER_LITER


def calculate_route(origin, destination, waypoints=None):
    """Calculate route using Google Directions API"""
    url = "https://maps.googleapis.com/maps/api/directions/json"

    params = {
        'origin': f"{origin['lat']},{origin['lng']}",
        'destination': f"{destination['lat']},{destination['lng']}",
        'key': GOOGLE_MAPS_API_KEY,
        'alternatives': 'true',
        'mode': 'driving'
    }

    if waypoints and len(waypoints) > 0:
        waypoint_str = "|".join([f"{point['lat']},{point['lng']}" for point in waypoints])
        params['waypoints'] = f"optimize:true|{waypoint_str}"

    try:
        response = requests.get(url, params=params)
        logging.debug(f"Directions Response: {response.text}")
        response.raise_for_status()
        data = response.json()

        routes = []
        if data['status'] == 'OK' and 'routes' in data:
            for i, route in enumerate(data['routes']):
                # Extract distance and duration
                distance_meters = 0
                duration_seconds = 0

                steps = []
                for leg in route['legs']:
                    distance_meters += leg['distance']['value']
                    duration_seconds += leg['duration']['value']

                    for step in leg['steps']:
                        steps.append({
                            'start_location': step['start_location'],
                            'end_location': step['end_location'],
                            'polyline': step['polyline']['points']
                        })

                # Calculate fuel cost
                distance_km = distance_meters / 1000
                fuel_cost = calculate_fuel_cost(distance_km)

                # Extract polyline for entire route
                overview_polyline = route['overview_polyline']['points']

                routes.append({
                    'polyline': overview_polyline,
                    'steps': steps,
                    'distance': {
                        'meters': distance_meters,
                        'text': f"{(distance_meters / 1000):.2f} km"
                    },
                    'duration': {
                        'seconds': duration_seconds,
                        'text': f"{(duration_seconds / 60):.0f} mins"
                    },
                    'fuel_cost': round(fuel_cost, 2)
                })

            return routes

        logging.warning(f"No routes found or API error: {data.get('status')}")
        return []

    except Exception as e:
        logging.error(f"Route calculation error: {e}")
        return []


@app.route('/cluster', methods=['POST'])
def cluster():
    """Perform clustering on delivery locations"""
    try:
        data = request.json
        locations = data.get('locations', [])
        warehouses = data.get('warehouses', [])
        radius = float(data.get('radius', 500))

        # Convert radius from meters to degrees (approximate)
        eps = radius / 111000

        # If no warehouses specified, use a default
        if not warehouses:
            warehouses = [{'lat': locations[0]['lat'], 'lng': locations[0]['lng']}]

        # Extract coordinates for clustering
        coords = np.array([[loc['lat'], loc['lng']] for loc in locations])

        # Perform DBSCAN clustering
        db = DBSCAN(eps=eps, min_samples=2, metric='haversine').fit(np.radians(coords))
        labels = db.labels_

        # Process clusters
        unique_labels = set(labels)
        clusters_info = []

        for label in unique_labels:
            mask = labels == label
            cluster_points = coords[mask]

            # Skip noise points (label = -1)
            if label != -1:
                # Calculate cluster center
                center_lat = float(np.mean(cluster_points[:, 0]))
                center_lng = float(np.mean(cluster_points[:, 1]))

                # Find closest warehouse to this cluster
                closest_warehouse = None
                min_distance = float('inf')

                for warehouse in warehouses:
                    # Simple Euclidean distance for demonstration
                    dist = ((warehouse['lat'] - center_lat) ** 2 +
                            (warehouse['lng'] - center_lng) ** 2) ** 0.5

                    if dist < min_distance:
                        min_distance = dist
                        closest_warehouse = warehouse

                # Create cluster info
                cluster_info = {
                    'label': int(label),
                    'center': {
                        'lat': center_lat,
                        'lng': center_lng
                    },
                    'point_count': int(np.sum(mask)),
                    'points': [{'lat': float(point[0]), 'lng': float(point[1])}
                               for point in cluster_points],
                    'warehouse': closest_warehouse
                }
                clusters_info.append(cluster_info)

        # Add cluster labels to the original locations
        clustered_locations = []
        for i, location in enumerate(locations):
            clustered_locations.append({
                'lat': location['lat'],
                'lng': location['lng'],
                'cluster': int(labels[i])
            })

        return jsonify({
            'locations': clustered_locations,
            'clusters': clusters_info,
            'warehouses': warehouses,
            'noise_points': int(np.sum(labels == -1)),
            'total_clusters': len(clusters_info)
        })

    except Exception as e:
        logging.error(f"Clustering error: {str(e)}")
        return jsonify({'error': str(e)}), 400


@app.route('/calculate_routes', methods=['POST'])
def calculate_routes():
    """Calculate optimal routes for clusters"""
    try:
        data = request.json
        clusters = data.get('clusters', [])

        if not clusters:
            return jsonify({'error': 'No clusters provided'}), 400

        results = []

        for cluster in clusters:
            warehouse = cluster.get('warehouse')
            center = cluster.get('center')
            points = cluster.get('points', [])

            if not warehouse or not points:
                continue

            # Calculate route from warehouse to all points in cluster
            routes = calculate_route(warehouse, center, points)

            # Get weather at cluster center
            weather = get_weather(center['lat'], center['lng'])

            results.append({
                'cluster_id': cluster.get('label'),
                'warehouse': warehouse,
                'cluster_center': center,
                'points': points,
                'routes': routes,
                'weather': weather
            })

        return jsonify({
            'success': True,
            'results': results
        })

    except Exception as e:
        logging.error(f"Route calculation error: {str(e)}")
        return jsonify({'error': str(e)}), 400


@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    """Google Places Autocomplete API wrapper"""
    query = request.args.get('query', '')
    if not query:
        return jsonify([])

    url = 'https://maps.googleapis.com/maps/api/place/autocomplete/json'
    params = {
        'input': query,
        'key': GOOGLE_MAPS_API_KEY,
        'types': 'geocode'
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        results = []
        if data['status'] == 'OK' and 'predictions' in data:
            for prediction in data['predictions']:
                results.append({
                    'label': prediction['description'],
                    'value': prediction['description'],
                    'place_id': prediction['place_id']
                })
        return jsonify(results)

    except Exception as e:
        logging.error(f"Autocomplete error: {e}")
        return jsonify([])


@app.route('/calculate_optimized_routes', methods=['POST'])
def calculate_optimized_routes():
    """First cluster locations, then calculate optimized routes"""
    try:
        data = request.json
        locations = data.get('locations', [])
        warehouses = data.get('warehouses', [])
        radius = float(data.get('radius', 500))

        # Step 1: Perform clustering
        cluster_data = {
            'locations': locations,
            'warehouses': warehouses,
            'radius': radius
        }

        # Call clustering logic directly instead of making a separate request
        clustering_result = cluster().get_json()

        if 'error' in clustering_result:
            return jsonify(clustering_result), 400

        # Step 2: Calculate routes for each cluster
        routes_data = {
            'clusters': clustering_result.get('clusters', [])
        }

        # Call route calculation logic directly
        routes_result = calculate_routes().get_json()

        if 'error' in routes_result:
            return jsonify(routes_result), 400

        # Combine results
        return jsonify({
            'success': True,
            'clustering': clustering_result,
            'routing': routes_result
        })

    except Exception as e:
        logging.error(f"Optimized routes error: {str(e)}")
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)