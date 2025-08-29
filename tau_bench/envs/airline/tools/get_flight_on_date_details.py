# Copyright Sierra

import json
from typing import Any, Dict
from tau_bench.envs.tool import Tool


class GetFlightOnDateDetails(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], flight_id: str, date:str) -> str:
        flights = data["flights"]
        if flight_id in flights:
            flight = flights[flight_id]
            fl_dates = flight.get('dates')
            if fl_dates:
                if date in fl_dates:
                    return json.dumps(fl_dates.get(date))
        return "Error: flight on date was not found"

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_flight_on_date_details",
                "description": "Get the details of a flight occurence in a specific date. For example, returns the flight status, available seats and prices.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "flight_id": {
                            "type": "string",
                            "description": "The flight id, such as 'HAT001'.",
                        },
                        "date": {
                            "type": "string",
                            "description": "The date of the flight, such as '2024-05-02'.",
                        },
                    },
                    "required": ["flight_id", "date"],
                },
            },
        }
