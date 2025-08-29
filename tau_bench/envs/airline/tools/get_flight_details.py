# Copyright Sierra

import copy
import json
from typing import Any, Dict
from tau_bench.envs.tool import Tool


class GetFlightDetails(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], flight_id: str) -> str:
        flights = data["flights"]
        if flight_id in flights:
            flight = copy.deepcopy(flights[flight_id])
            
            if flight.get('dates'):
                del flight['dates']
            return json.dumps(flight)
        return "Error: flight not found"

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_flight_details",
                "description": "Get the details of a flight. For example, returns the flight origin, destination, scheduled detaprture and arrival times",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "flight_id": {
                            "type": "string",
                            "description": "The flight id, such as 'HAT001'.",
                        },
                    },
                    "required": ["flight_id"],
                },
            },
        }
