import logging
import os
import json
from typing import List
from pathlib import Path

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

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# Use absolute path for Windows
COURSE_CONTENT_FILE = r"D:\MURFAI\ten-days-of-voice-agents-2025\shared-data\day4_tutor_content.json"

class ActiveRecallCoach(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are an Active Recall Coach that helps users learn through teaching. You have three learning modes:

LEARNING MODES:
1. LEARN MODE (Voice: "Matthew"): Explain concepts clearly and concisely using the course content.
2. QUIZ MODE (Voice: "Alicia"): Ask questions to test understanding using sample questions.
3. TEACH_BACK MODE (Voice: "Ken"): Ask users to explain concepts back and provide basic qualitative feedback.

HOW TO RESPOND:
- Always start by checking the current mode and available concepts
- In LEARN mode: Use the concept summary to explain clearly
- In QUIZ mode: Use the sample_question to test understanding  
- In TEACH_BACK mode: Ask the user to explain the concept back to you
- Always allow mode switching when requested
- Be encouraging and educational

Use your tools to load content and switch modes as needed.
"""
        )
        self.current_mode = "learn"
        self.concepts = self._load_concepts()
        logger.info(f"Loaded {len(self.concepts)} concepts")

    def _load_concepts(self):
        """Load concepts on initialization"""
        try:
            logger.info(f"Looking for course content at: {COURSE_CONTENT_FILE}")
            
            if os.path.exists(COURSE_CONTENT_FILE):
                with open(COURSE_CONTENT_FILE, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                logger.info(f"Successfully loaded {len(content)} concepts")
                return content
            else:
                logger.error(f"File not found: {COURSE_CONTENT_FILE}")
                # Try relative path as fallback
                relative_path = "./shared-data/day4_tutor_content.json"
                if os.path.exists(relative_path):
                    with open(relative_path, 'r', encoding='utf-8') as f:
                        content = json.load(f)
                    logger.info(f"Loaded from relative path: {len(content)} concepts")
                    return content
                return []
        except Exception as e:
            logger.error(f"Error loading concepts: {e}")
            return []

    @function_tool()
    async def load_course_content(self, context: RunContext) -> str:
        """Load and display available course concepts."""
        if not self.concepts:
            return "No course content available. Please check the content file."
        
        concept_list = []
        for concept in self.concepts:
            concept_list.append(f"• {concept.get('title')} ({concept.get('id')})")
        
        return "Available concepts:\n" + "\n".join(concept_list)

    @function_tool()
    async def switch_mode(self, context: RunContext, new_mode: str) -> str:
        """Switch between learning modes: learn, quiz, or teach_back."""
        valid_modes = ["learn", "quiz", "teach_back"]
        if new_mode.lower() not in valid_modes:
            return f"Invalid mode. Please choose from: {', '.join(valid_modes)}"
        
        old_mode = self.current_mode
        self.current_mode = new_mode.lower()
        
        mode_descriptions = {
            "learn": "Learn mode - I'll explain concepts (Voice: Matthew)",
            "quiz": "Quiz mode - I'll ask you questions (Voice: Alicia)", 
            "teach_back": "Teach-back mode - You explain concepts to me (Voice: Ken)"
        }
        
        return f"Switched to {mode_descriptions[self.current_mode]}"

    @function_tool()
    async def explain_concept(self, context: RunContext, concept_name: str = None) -> str:
        """Explain a concept in learn mode."""
        if not self.concepts:
            return "No concepts available to explain."
        
        concept = None
        if concept_name:
            # Find concept by ID or title
            for c in self.concepts:
                if c.get('id') == concept_name.lower() or c.get('title').lower() == concept_name.lower():
                    concept = c
                    break
        
        if not concept:
            # Use first concept as default
            concept = self.concepts[0]
        
        return f"Let me explain {concept.get('title')}:\n\n{concept.get('summary')}"

    @function_tool()
    async def quiz_concept(self, context: RunContext, concept_name: str = None) -> str:
        """Quiz a concept in quiz mode."""
        if not self.concepts:
            return "No concepts available for quiz."
        
        concept = None
        if concept_name:
            for c in self.concepts:
                if c.get('id') == concept_name.lower() or c.get('title').lower() == concept_name.lower():
                    concept = c
                    break
        
        if not concept:
            concept = self.concepts[0]
        
        return f"Quiz question about {concept.get('title')}:\n\n{concept.get('sample_question')}"

    @function_tool()
    async def teach_back_concept(self, context: RunContext, concept_name: str = None) -> str:
        """Ask user to explain a concept back in teach-back mode."""
        if not self.concepts:
            return "No concepts available for teach-back."
        
        concept = None
        if concept_name:
            for c in self.concepts:
                if c.get('id') == concept_name.lower() or c.get('title').lower() == concept_name.lower():
                    concept = c
                    break
        
        if not concept:
            concept = self.concepts[0]
        
        return f"Now it's your turn to teach me about {concept.get('title')}! Please explain this concept in your own words."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Initialize the agent
    agent = ActiveRecallCoach()
    
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