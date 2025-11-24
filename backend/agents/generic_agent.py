# backend/agents/generic_agent.py

def classify_domain(description: str) -> str:
    """
    Very simple rule-based domain classifier.
    Later we'll replace/add a Gemini-based version.
    """
    text = description.lower()

    network_keywords = ["wifi", "wi-fi", "internet", "latency", "network", "vpn"]
    access_keywords = ["login", "password", "access", "account", "locked"]
    billing_keywords = ["bill", "billing", "invoice", "payment", "charge", "refund"]

    if any(k in text for k in network_keywords):
        return "network"
    if any(k in text for k in access_keywords):
        return "access"
    if any(k in text for k in billing_keywords):
        return "billing"
    return "generic"


def suggest_resolution(description: str, domain: str) -> dict:
    """
    Stub 'AI' suggestion.
    Later this will call Gemini via ADK.
    Returns a dict with agent_name, resolution text, confidence score.
    """
    agent_name = "generic_agent"

    if domain == "network":
        resolution = (
            "Please restart the router and check the network cables. "
            "If the issue persists, try connecting to a different access point."
        )
        confidence = 0.7
    elif domain == "access":
        resolution = (
            "Reset the user's password and ensure their account is not locked. "
            "Verify MFA settings if enabled."
        )
        confidence = 0.7
    elif domain == "billing":
        resolution = (
            "Check the billing system for duplicate charges. "
            "If confirmed, initiate a refund and notify the user."
        )
        confidence = 0.6
    else:
        resolution = (
            "Please collect more details from the user about the issue and "
            "check relevant logs or systems before proceeding."
        )
        confidence = 0.4

    return {
        "agent_name": agent_name,
        "resolution": resolution,
        "confidence": confidence,
    }
