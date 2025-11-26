import logging
import json
import os
from typing import Dict, Any
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    metrics,
    tokenize,
    function_tool,
    cli,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("fraud-agent")

load_dotenv(".env.local")

# Fraud database file - using relative path to shared-data
PROJECT_ROOT = Path(__file__).parent.parent.parent
FRAUD_DB_FILE = PROJECT_ROOT / "shared-data" / "fraud_cases.json"

class FraudAlertAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are a professional fraud detection representative for SecureBank. Your role is to handle fraud alert calls.

CRITICAL RULES:
- NEVER ask for full card numbers, PINs, passwords, or sensitive credentials
- Use only the security question from the database for verification
- Speak in a calm, professional, and reassuring manner
- If verification fails, end the call politely
- Clearly explain what action will be taken based on customer response

CALL FLOW:
1. Greet and introduce yourself as SecureBank Fraud Department
2. Ask for the customer by name (from database)
3. Ask the security question exactly as stored in database
4. If correct answer: proceed to transaction review
5. If incorrect answer: politely end the call
6. Describe the suspicious transaction details
7. Ask if they made this transaction (yes/no)
8. Based on answer: mark as safe or fraudulent
9. Explain what action is being taken
10. End call professionally

ALWAYS follow this exact sequence and use professional, reassuring language.
"""
        )
        self.fraud_db = self._load_fraud_db()
        self.current_case = None
        self.verification_passed = False
        self.transaction_reviewed = False

    def _load_fraud_db(self) -> Dict[str, Any]:
        """Load fraud cases from database"""
        try:
            if FRAUD_DB_FILE.exists():
                with open(FRAUD_DB_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"Successfully loaded fraud database with {len(data.get('fraud_cases', []))} cases")
                return data
            else:
                logger.error(f"Fraud database file not found: {FRAUD_DB_FILE}")
                # Create default structure if file doesn't exist
                default_data = {"fraud_cases": []}
                with open(FRAUD_DB_FILE, 'w', encoding='utf-8') as f:
                    json.dump(default_data, f, indent=2)
                return default_data
        except Exception as e:
            logger.error(f"Error loading fraud database: {e}")
            return {"fraud_cases": []}

    def _save_fraud_db(self):
        """Save updated fraud cases to database"""
        try:
            with open(FRAUD_DB_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.fraud_db, f, indent=2, ensure_ascii=False)
            logger.info("Fraud database updated successfully")
        except Exception as e:
            logger.error(f"Error saving fraud database: {e}")

    @function_tool()
    async def load_fraud_case(self, context: RunContext, username: str) -> str:
        """Load a fraud case for the specified user"""
        for case in self.fraud_db.get("fraud_cases", []):
            if case.get("userName", "").lower() == username.lower() and case.get("case") == "pending_review":
                self.current_case = case
                logger.info(f"Loaded fraud case for user: {username}")
                return f"Found pending fraud case for {username}. Ready for verification."
        
        return f"No pending fraud cases found for {username}. Please check the username and try again."

    @function_tool()
    async def verify_customer(self, context: RunContext, answer: str) -> str:
        """Verify customer identity using security question"""
        if not self.current_case:
            return "No fraud case loaded. Please load a fraud case first."
        
        correct_answer = self.current_case.get("securityAnswer", "").lower()
        user_answer = answer.lower()
        
        if user_answer == correct_answer:
            self.verification_passed = True
            return "Verification successful. I can now proceed with the transaction review."
        else:
            self.verification_passed = False
            # Update case status
            self.current_case["case"] = "verification_failed"
            self.current_case["outcome"] = f"Customer failed verification on {datetime.now().isoformat()}"
            self._save_fraud_db()
            return "Verification failed. For security reasons, I cannot proceed with this call. Please contact our customer service department for assistance. Goodbye."

    @function_tool()
    async def describe_transaction(self, context: RunContext) -> str:
        """Describe the suspicious transaction details"""
        if not self.current_case:
            return "No fraud case loaded."
        
        if not self.verification_passed:
            return "Customer verification required before describing transaction."
        
        case = self.current_case
        description = f"""
I'm calling about a suspicious transaction on your card ending in {case['cardEnding']}. 
We noticed a transaction for {case['transactionAmount']} at {case['transactionName']} 
({case['transactionSource']}) on {case['transactionTime']} from {case['location']}. 
This was categorized as {case['transactionCategory']}.
"""
        self.transaction_reviewed = True
        return description

    @function_tool()
    async def confirm_transaction(self, context: RunContext, customer_response: str) -> str:
        """Ask customer to confirm if they made the transaction"""
        if not self.verification_passed:
            return "Customer verification required."
        
        if not self.transaction_reviewed:
            return "Please describe the transaction first."
        
        response_lower = customer_response.lower()
        
        if "yes" in response_lower or "yeah" in response_lower or "i did" in response_lower or "confirm" in response_lower:
            # Mark as safe
            self.current_case["case"] = "confirmed_safe"
            self.current_case["outcome"] = f"Customer confirmed transaction as legitimate on {datetime.now().isoformat()}"
            self._save_fraud_db()
            return "Thank you for confirming. We'll mark this transaction as legitimate and no further action is needed. Your card remains active. Thank you for your time and helping us keep your account secure."
        
        elif "no" in response_lower or "nope" in response_lower or "not me" in response_lower or "fraud" in response_lower:
            # Mark as fraudulent
            self.current_case["case"] = "confirmed_fraud"
            self.current_case["outcome"] = f"Customer denied transaction - marked as fraudulent on {datetime.now().isoformat()}"
            self._save_fraud_db()
            return "Thank you for confirming this was not your transaction. We're immediately blocking your card to prevent further unauthorized use and initiating a dispute process. A new card will be mailed to you within 3-5 business days. Please check your email for further instructions. Thank you for your prompt attention to this security matter."
        
        else:
            return "I apologize, I didn't understand. Could you please confirm if you made this transaction? Please answer yes or no."

    @function_tool()
    async def get_security_question(self, context: RunContext) -> str:
        """Get the security question for the current fraud case"""
        if not self.current_case:
            return "No fraud case loaded."
        
        return self.current_case.get("securityQuestion", "No security question available.")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Initialize the fraud agent
    agent = FraudAlertAgent()
    
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))