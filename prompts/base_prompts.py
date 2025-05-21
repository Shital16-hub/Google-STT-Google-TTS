{
  "base_system_prompt": {
    "context": "You are a professional roadside assistance agent focused on providing efficient, clear assistance.",
    "template": "Respond in a natural, conversational manner while maintaining professionalism. Focus on gathering necessary information efficiently and providing clear, actionable responses. {context}"
  },
  
  "base_greeting": {
    "context": "Initial greeting for any service type.",
    "template": "Hello! I'm here to help with your roadside assistance needs. {confirmations}How can I assist you today?"
  },
  
  "base_collecting_name": {
    "context": "Standard name collection for any service.",
    "template": "{confirmations}Could I get your name please?"
  },
  
  "base_collecting_phone": {
    "context": "Standard phone collection for any service.",
    "template": "{confirmations}What's the best phone number to reach you at?"
  },
  
  "base_collecting_location": {
    "context": "Standard location collection for any service.",
    "template": "{confirmations}Could you tell me your current location? Please include any helpful landmarks or details."
  },
  
  "base_safety_check": {
    "context": "Standard safety check for any service.",
    "template": "First, I want to make sure you're safe. Are you in a secure location away from traffic?"
  },
  
  "base_confirmation": {
    "context": "Standard confirmation template.",
    "template": "Let me confirm what I have so far: {confirmations}Is that all correct?"
  },
  
  "base_correction": {
    "context": "Handle corrections to collected information.",
    "template": "I apologize for the misunderstanding. Let me correct that information. {correction_prompt}"
  },
  
  "base_handoff": {
    "context": "Standard handoff to dispatch.",
    "template": "I'll need to transfer you to our dispatch team who can better assist with your situation. {handoff_reason}"
  },
  
  "base_wait_time": {
    "context": "Standard wait time notification.",
    "template": "Based on current conditions, the estimated wait time is {wait_time} minutes. I'll make sure to keep you updated on any changes."
  },
  
  "base_status_update": {
    "context": "Standard status update template.",
    "template": "Here's an update on your service: {status_message}"
  },
  
  "base_completion": {
    "context": "Standard service completion.",
    "template": "Is there anything else you need assistance with?"
  },
  
  "base_emergency": {
    "context": "Handle emergency situations.",
    "template": "I understand this is an emergency. Your safety is our top priority. {emergency_instructions}"
  },
  
  "base_weather_delay": {
    "context": "Handle weather-related delays.",
    "template": "Due to current weather conditions, our service times may be longer than usual. Your safety is our priority, and we'll get to you as quickly as we safely can."
  },
  
  "base_after_hours": {
    "context": "Handle after-hours service requests.",
    "template": "Please note that since this is after normal business hours, there will be an additional fee of ${after_hours_fee}."
  },
  
  "base_clarification": {
    "context": "Request clarification on unclear information.",
    "template": "Could you please clarify {clarification_point}? This will help me better assist you."
  },
  
  "base_service_requirements": {
    "context": "Explain service requirements.",
    "template": "For this service, we'll need: {requirements}. Do you have any questions about these requirements?"
  },
  
  "base_price_explanation": {
    "context": "Explain service pricing.",
    "template": "Let me break down the pricing for you:\n- Base service: ${base_price}\n{additional_fees}Total cost: ${total_price}\nWould you like to proceed?"
  },
  
  "base_feedback": {
    "context": "Request service feedback.",
    "template": "Thank you for using our service. Would you mind taking a moment to rate your experience? This helps us improve our service."
  },
  
  "base_contact_preference": {
    "context": "Establish contact preferences.",
    "template": "How would you prefer to receive updates about your service - call, text, or both?"
  },
  
  "base_callback": {
    "context": "Handle callback requests.",
    "template": "I'll have someone call you back within {callback_time} minutes. Is this number {phone_number} the best one to reach you?"
  }
}