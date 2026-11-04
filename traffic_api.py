
import requests


def get_tomtom_traffic(latitude, longitude, api_key):
    """
    Fetch real-time traffic flow data from TomTom.
    """

    if not api_key or api_key == "PASTE_YOUR_API_KEY_HERE":
        return {
            "success": False,
            "message": "TomTom API key is not configured."
        }

    url = (
        "https://api.tomtom.com/traffic/services/4/"
        "flowSegmentData/absolute/10/json"
    )

    params = {
        "point": f"{latitude},{longitude}",
        "key": api_key
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        response.raise_for_status()
        data = response.json()

        flow = data.get("flowSegmentData", {})

        return {
            "success": True,
            "current_speed": flow.get("currentSpeed"),
            "free_flow_speed": flow.get("freeFlowSpeed"),
            "current_travel_time": flow.get("currentTravelTime"),
            "free_flow_travel_time": flow.get("freeFlowTravelTime"),
            "confidence": flow.get("confidence"),
            "road_closure": flow.get("roadClosure"),
            "message": "Live traffic data received successfully."
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "message": "TomTom API request timed out."
        }

    except requests.exceptions.HTTPError as error:
        return {
            "success": False,
            "message": f"TomTom HTTP error: {error}"
        }

    except requests.exceptions.RequestException as error:
        return {
            "success": False,
            "message": f"Connection error: {error}"
        }

    except (ValueError, KeyError) as error:
        return {
            "success": False,
            "message": f"Invalid API response: {error}"
        }
