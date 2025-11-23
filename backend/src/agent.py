import logging
import os
import json
import datetime
from typing import Optional, List
from pathlib import Path

from dotenv import load_dotenv
from livekit import rtc
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

# JSON file for data persistence
WELLNESS_LOG_FILE = "wellness_log.json"

class WellnessCompanion(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are a supportive, realistic, and grounded health and wellness companion. Your role is to provide daily check-ins that are encouraging but practical.

DAILY CHECK-IN WORKFLOW:
1. GREETING & MOOD CHECK:
   - Start with a warm, friendly greeting
   - Ask about their current mood and energy level
   - Examples: "How are you feeling today?", "What's your energy like?", "Anything on your mind right now?"

2. DAILY INTENTIONS:
   - Ask about 1-3 practical goals for the day
   - Examples: "What are 1-3 things you'd like to accomplish today?", "Is there anything you want to do for yourself today?"
   - Keep goals simple and achievable

3. SUPPORTIVE REFLECTION:
   - Offer simple, realistic advice or reflections
   - Suggestions should be small, actionable, and non-medical
   - Examples: "Breaking large tasks into smaller steps can help", "Remember to take short breaks", "A 5-minute walk might help clear your mind"

4. RECAP & CLOSE:
   - Summarize today's mood and main objectives
   - Confirm with: "Does this sound right to you?"
   - End with an encouraging note

IMPORTANT RULES:
- NEVER provide medical advice or diagnosis
- Keep responses supportive but realistic
- Reference previous check-ins when relevant
- Store all check-in data using the save_checkin tool
- Use load_previous_checkins to get historical data
- Keep conversations brief and focused (3-5 minutes max)
"""
        )

    @function_tool()
    async def save_checkin(
        self,
        context: RunContext,
        mood: str,
        energy_level: str,
        objectives: List[str],
        summary: str = ""
    ) -> str:
        """
        Save today's wellness check-in to the JSON log file.
        
        Call this at the end of each check-in conversation.
        """
        # Ensure the log file exists
        log_file = Path(WELLNESS_LOG_FILE)
        
        # Load existing data or create empty list
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                data = []
        else:
            data = []
        
        # Create new entry
        new_entry = {
            "date": datetime.datetime.now().isoformat(),
            "mood": mood,
            "energy_level": energy_level,
            "objectives": objectives,
            "summary": summary
        }
        
        # Add to data and save
        data.append(new_entry)
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return f"Check-in saved successfully. You now have {len(data)} entries in your wellness log."

    @function_tool()
    async def load_previous_checkins(self, context: RunContext, days_back: int = 7) -> str:
        """
        Load previous wellness check-ins from the JSON file.
        
        Use this to reference past conversations and provide continuity.
        """
        log_file = Path(WELLNESS_LOG_FILE)
        
        if not log_file.exists():
            return "No previous check-ins found. This appears to be your first session!"
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            return "Error reading wellness log. Starting fresh for today."
        
        if not data:
            return "No previous check-ins found in the log."
        
        # Get recent entries (last N days)
        recent_entries = []
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_back)
        
        for entry in data[-10:]:  # Check last 10 entries max
            try:
                entry_date = datetime.datetime.fromisoformat(entry["date"])
                if entry_date >= cutoff_date:
                    recent_entries.append(entry)
            except (KeyError, ValueError):
                continue
        
        if not recent_entries:
            return f"No check-ins found from the last {days_back} days."
        
        # Format the response
        response = f"Found {len(recent_entries)} check-in(s) from the last {days_back} days:\n\n"
        
        for i, entry in enumerate(recent_entries[-3:], 1):  # Show last 3 entries
            entry_date = datetime.datetime.fromisoformat(entry["date"]).strftime("%b %d")
            response += f"• {entry_date}: Mood: {entry.get('mood', 'N/A')}, Energy: {entry.get('energy_level', 'N/A')}\n"
            if entry.get('objectives'):
                response += f"  Objectives: {', '.join(entry['objectives'][:2])}\n"
        
        return response

    @function_tool()
    async def get_weekly_insights(self, context: RunContext) -> str:
        """
        Provide basic insights from the past week's check-ins.
        
        Use this when the user asks about trends or patterns.
        """
        log_file = Path(WELLNESS_LOG_FILE)
        
        if not log_file.exists():
            return "Not enough data yet for weekly insights. Check back after a few days!"
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            return "Could not read wellness data for insights."
        
        if len(data) < 2:
            return "Need more check-in data to provide insights. Keep going!"
        
        # Analyze last 7 days
        week_ago = datetime.datetime.now() - datetime.timedelta(days=7)
        recent_data = []
        
        for entry in data:
            try:
                entry_date = datetime.datetime.fromisoformat(entry["date"])
                if entry_date >= week_ago:
                    recent_data.append(entry)
            except (KeyError, ValueError):
                continue
        
        if len(recent_data) < 2:
            return "Not enough recent data for weekly insights."
        
        # Simple analysis
        mood_counts = {}
        energy_counts = {}
        objectives_count = 0
        days_with_objectives = 0
        
        for entry in recent_data:
            mood = entry.get('mood', '').lower()
            energy = entry.get('energy_level', '').lower()
            objectives = entry.get('objectives', [])
            
            if mood:
                mood_counts[mood] = mood_counts.get(mood, 0) + 1
            if energy:
                energy_counts[energy] = energy_counts.get(energy, 0) + 1
            
            if objectives:
                objectives_count += len(objectives)
                days_with_objectives += 1
        
        # Build insights
        insights = []
        
        if mood_counts:
            common_mood = max(mood_counts.items(), key=lambda x: x[1])
            insights.append(f"Your most common mood this week was '{common_mood[0]}'")
        
        if energy_counts:
            common_energy = max(energy_counts.items(), key=lambda x: x[1])
            insights.append(f"You typically felt '{common_energy[0]}' energy levels")
        
        if days_with_objectives > 0:
            avg_objectives = objectives_count / days_with_objectives
            insights.append(f"You set about {avg_objectives:.1f} goals per day on average")
        
        if not insights:
            return "I notice you've been consistent with your check-ins. That's great for building awareness!"
        
        return "Here's what I noticed from your recent check-ins:\n" + "\n".join(f"• {insight}" for insight in insights)


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Initialize wellness companion session
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(
            model="gemini-2.5-flash",
        ),
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
    session.userdata = {}

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

    # Start the wellness companion session
    await session.start(
        agent=WellnessCompanion(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))