FLIGHT_CANCELLATION_POLICY = f"""
Here is the policy:
1. Confirm which flight the customer is asking to cancel.
1a) If the customer is asking about the same flight, proceed to next step.
1b) If the customer is not, call 'escalate_to_human' function.
2. Confirm if the customer wants a refund or flight credits.
3. If the customer wants a refund follow step 3a). If the customer wants flight credits move to step 4.
3a) Call the initiate_refund function.
3b) Inform the customer that the refund will be processed within 3-5 business days.
4. If the customer wants flight credits, call the initiate_flight_credits function.
4a) Inform the customer that the flight credits will be available in the next 15 minutes.
5. If the customer has no further questions, call the case_resolved function.
"""


FLIGHT_CHANGE_POLICY = f"""
Here is the policy:
1. Verify the flight details and the reason for the change request.
2. Call valid_to_change_flight function:
2a) If the flight is confirmed valid to change: proceed to the next step.
2b) If the flight is not valid to change: politely let the customer know they cannot change their flight.
3. Suggest an flight one day earlier to customer.
4. Check for availability on the requested new flight:
4a) If seats are available, proceed to the next step.
4b) If seats are not available, offer alternative flights or advise the customer to check back later.
5. Inform the customer of any fare differences or additional charges.
6. Call the change_flight function.
7. If the customer has no further questions, call the case_resolved function.
"""
