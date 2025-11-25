import logging
import os
import json
from typing import List, Dict, Any
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
    inference,
    cli,
    metrics,
    tokenize,
    function_tool,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("sdr-agent")

load_dotenv(".env.local")

# Company data file
COMPANY_DATA_FILE = r"D:\MURFAI\ten-days-of-voice-agents-2025\shared-data\company_faq.json"
LEADS_FILE = r"D:\MURFAI\ten-days-of-voice-agents-2025\shared-data\leads.json"

class SDRVoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are a friendly Sales Development Representative (SDR) for an Indian startup. Your goal is to:

1. GREET warmly and introduce yourself as an SDR
2. UNDERSTAND their needs by asking what brought them here and what they're working on
3. ANSWER questions about the company using the FAQ content
4. COLLECT lead information naturally during the conversation
5. SUMMARIZE at the end and store the lead details

KEY BEHAVIORS:
- Be warm, professional, and conversational
- Ask open-ended questions to understand their needs
- Use the FAQ tool to answer company/product questions accurately
- Naturally collect lead info: name, company, email, role, use case, team size, timeline
- When user says they're done (e.g., "that's all", "thanks", "I'm done"), provide a summary and end the call
- Store all collected information in the lead database

CONVERSATION FLOW:
1. Warm greeting and introduction
2. Ask about their needs and current work
3. Answer any questions using FAQ
4. Collect lead information naturally
5. Provide summary and close when done
"""
        )
        self.company_data = self._load_company_data()
        self.lead_data = self._initialize_lead_data()
        self.conversation_ended = False

    def _load_company_data(self) -> Dict[str, Any]:
        """Load company FAQ and information"""
        try:
            logger.info(f"Looking for company data at: {COMPANY_DATA_FILE}")
            
            if os.path.exists(COMPANY_DATA_FILE):
                with open(COMPANY_DATA_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"Successfully loaded company data for: {data.get('company_name', 'Unknown')}")
                return data
            else:
                logger.error(f"Company data file not found: {COMPANY_DATA_FILE}")
                # Create default structure if file doesn't exist
                return {
                    "company_name": "ZapScale",
                    "faq": []
                }
        except Exception as e:
            logger.error(f"Error loading company data: {e}")
            return {"company_name": "ZapScale", "faq": []}

    def _initialize_lead_data(self) -> Dict[str, Any]:
        """Initialize lead data with default values"""
        return {
            "name": "Not provided",
            "company": "Not provided", 
            "email": "Not provided",
            "role": "Not provided",
            "use_case": "Not provided",
            "team_size": "Not provided",
            "timeline": "Not provided",
            "conversation_date": datetime.now().isoformat()
        }

    def _save_lead(self):
        """Save lead data to JSON file"""
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(LEADS_FILE), exist_ok=True)
            
            # Load existing leads or create empty list
            existing_leads = []
            if os.path.exists(LEADS_FILE):
                try:
                    with open(LEADS_FILE, 'r', encoding='utf-8') as f:
                        existing_leads = json.load(f)
                    if not isinstance(existing_leads, list):
                        existing_leads = []
                except (json.JSONDecodeError, Exception):
                    existing_leads = []
            
            # Add new lead
            existing_leads.append(self.lead_data)
            
            # Save back to file
            with open(LEADS_FILE, 'w', encoding='utf-8') as f:
                json.dump(existing_leads, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Lead saved successfully: {self.lead_data}")
        except Exception as e:
            logger.error(f"Error saving lead: {e}")

    @function_tool()
    async def get_company_info(self, context: RunContext) -> str:
        """Get basic company information"""
        company_name = self.company_data.get("company_name", "Our company")
        description = self.company_data.get("description", "")
        
        if description:
            return f"{company_name}: {description}"
        else:
            return f"Welcome to {company_name}! How can I help you today?"

    @function_tool()
    async def search_faq(self, context: RunContext, question: str) -> str:
        """Search FAQ for answers to company, product, or pricing questions"""
        faq_entries = self.company_data.get("faq", [])
        
        if not faq_entries:
            return "I apologize, but I don't have the FAQ information available at the moment. Please visit our website for more details."
        
        # Simple keyword matching
        question_lower = question.lower()
        matched_entries = []
        
        for entry in faq_entries:
            question_text = entry.get("question", "").lower()
            answer_text = entry.get("answer", "").lower()
            
            # Check if any keywords from the question match FAQ
            keywords = ["what", "how", "who", "when", "where", "why", "pricing", "price", "cost", "free", "tier", "do", "does", "can"]
            question_words = set(question_lower.split())
            faq_words = set((question_text + " " + answer_text).split())
            
            # Simple overlap matching
            overlap = question_words.intersection(faq_words)
            if len(overlap) >= 2:  # At least 2 common words
                matched_entries.append(entry)
            elif any(keyword in question_lower for keyword in keywords):
                # If it's a proper question, include relevant entries
                for keyword in ["pricing", "price", "cost"]:
                    if keyword in question_lower and keyword in (question_text + answer_text):
                        matched_entries.append(entry)
                        break
        
        # If no matches found, return all FAQ or a default response
        if not matched_entries:
            return "I'd be happy to tell you about our company! We offer innovative solutions for businesses. Could you be more specific about what you'd like to know?"
        
        # Return the first matched entry
        best_match = matched_entries[0]
        return f"{best_match.get('question')}\n\n{best_match.get('answer')}"

    @function_tool()
    async def collect_lead_info(self, context: RunContext, field: str, value: str) -> str:
        """Collect lead information during conversation"""
        field_mapping = {
            "name": "name",
            "company": "company", 
            "email": "email",
            "role": "role",
            "use case": "use_case",
            "use_case": "use_case",
            "team size": "team_size", 
            "team_size": "team_size",
            "timeline": "timeline"
        }
        
        field_lower = field.lower()
        if field_lower in field_mapping:
            field_key = field_mapping[field_lower]
            self.lead_data[field_key] = value
            logger.info(f"Collected lead info: {field_key} = {value}")
            return f"Thanks! I've noted your {field}."
        else:
            # Store unknown fields in a notes section
            if "notes" not in self.lead_data:
                self.lead_data["notes"] = []
            self.lead_data["notes"].append(f"{field}: {value}")
            return f"I'll make a note of that information about {field}."

    @function_tool()
    async def end_conversation(self, context: RunContext) -> str:
        """End the conversation and save lead summary"""
        self.conversation_ended = True
        
        # Create summary using safe dictionary access
        summary_parts = []
        fields_to_summarize = [
            ("name", "Name"),
            ("company", "Company"), 
            ("role", "Role"),
            ("use_case", "Use Case"),
            ("team_size", "Team Size"),
            ("timeline", "Timeline")
        ]
        
        for field_key, display_name in fields_to_summarize:
            value = self.lead_data.get(field_key)
            if value and value != "Not provided":
                summary_parts.append(f"{display_name}: {value}")
        
        summary = "Great speaking with you! "
        if summary_parts:
            summary += "Here's a quick summary: " + "; ".join(summary_parts)
        else:
            summary += "Thank you for your interest in our company!"
        
        summary += " One of our team members will follow up with you soon. Have a great day!"
        
        # Save the lead
        try:
            self._save_lead()
            logger.info("Lead saved successfully in end_conversation")
        except Exception as e:
            logger.error(f"Failed to save lead in end_conversation: {e}")
        
        return summary


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Initialize the agent
    agent = SDRVoiceAgent()
    
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