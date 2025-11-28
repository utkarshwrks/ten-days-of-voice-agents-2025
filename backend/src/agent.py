import logging
import json
import random
from typing import Dict, Any, List
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

logger = logging.getLogger("game-master-agent")

load_dotenv(".env.local")

class GameMasterAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are a dramatic and immersive Game Master running a fantasy adventure in the world of Eldoria, a land of ancient magic, dragons, and forgotten kingdoms.

IMPORTANT RULES:
- Maintain a dramatic, engaging tone throughout the adventure
- Describe scenes vividly with sensory details (sights, sounds, smells)
- Always end your descriptions with a prompt for player action ("What do you do?", "How do you proceed?", "What's your next move?")
- Remember the player's past decisions and incorporate them into the story
- Create consequences for player choices
- Keep the story moving forward with appropriate challenges and discoveries

CURRENT ADVENTURE: "The Crystal of Aethel"
- The player is a novice adventurer in the village of Oakhaven
- They've been tasked with investigating strange occurrences in the Whispering Woods
- Rumors speak of an ancient artifact - the Crystal of Aethel - hidden in the woods

GAME MASTER GUIDELINES:
1. Start by setting the scene in Oakhaven village at dawn
2. Introduce the quest from Elder Malakai
3. Guide the player through the Whispering Woods with encounters and choices
4. Include at least 3 key encounters before reaching the crystal
5. Create a meaningful climax and resolution
6. Always maintain continuity - remember NPCs, items, and decisions

STORY BEATS:
- Village introduction and quest assignment
- Forest entrance and first encounter
- Ancient ruins puzzle
- Guardian confrontation
- Crystal discovery and choice

Remember: You are the eyes and ears of the player. Describe everything they experience!
"""
        )
        self.world_state = self._initialize_world_state()
        self.game_active = True

    def _initialize_world_state(self) -> Dict[str, Any]:
        """Initialize the game world state"""
        return {
            "player": {
                "name": "Adventurer",
                "health": 100,
                "max_health": 100,
                "inventory": ["torch", "provisions", "map"],
                "gold": 10,
                "location": "oakhaven_village",
                "reputation": 0
            },
            "npcs": {
                "elder_malakai": {
                    "name": "Elder Malakai",
                    "location": "oakhaven_village",
                    "attitude": "friendly",
                    "quest_given": False
                },
                "forest_guardian": {
                    "name": "Ancient Guardian",
                    "location": "ancient_ruins",
                    "attitude": "neutral",
                    "encountered": False
                }
            },
            "locations": {
                "oakhaven_village": {
                    "name": "Oakhaven Village",
                    "description": "A quiet village at the edge of Whispering Woods",
                    "visited": False
                },
                "whispering_woods": {
                    "name": "Whispering Woods",
                    "description": "A mysterious forest where trees seem to whisper secrets",
                    "visited": False
                },
                "ancient_ruins": {
                    "name": "Ancient Ruins",
                    "description": "Crumbling stone structures from a forgotten civilization",
                    "visited": False
                }
            },
            "quests": {
                "find_crystal": {
                    "name": "Find the Crystal of Aethel",
                    "status": "active",  # active, completed, failed
                    "description": "Investigate strange occurrences in Whispering Woods and find the legendary crystal"
                }
            },
            "events": {
                "elder_met": False,
                "forest_entered": False,
                "ruins_discovered": False,
                "crystal_found": False
            },
            "game_state": {
                "turn_count": 0,
                "current_chapter": 1
            }
        }

    def _dice_roll(self, sides: int = 20, modifier: int = 0) -> int:
        """Make a dice roll for game mechanics"""
        return random.randint(1, sides) + modifier

    def _update_location(self, new_location: str):
        """Update player location and trigger location-based events"""
        old_location = self.world_state["player"]["location"]
        self.world_state["player"]["location"] = new_location
        
        # Mark location as visited
        if new_location in self.world_state["locations"]:
            self.world_state["locations"][new_location]["visited"] = True
        
        # Trigger location-specific events
        if new_location == "whispering_woods" and not self.world_state["events"]["forest_entered"]:
            self.world_state["events"]["forest_entered"] = True
        elif new_location == "ancient_ruins" and not self.world_state["events"]["ruins_discovered"]:
            self.world_state["events"]["ruins_discovered"] = True

    @function_tool()
    async def roll_dice(self, context: RunContext, sides: int = 20, modifier: int = 0) -> str:
        """Roll dice for game mechanics - used by the GM for skill checks"""
        roll = self._dice_roll(sides, modifier)
        return f"Dice roll: {roll} (d{sides} + {modifier})"

    @function_tool()
    async def check_inventory(self, context: RunContext) -> str:
        """Check the player's current inventory and status"""
        player = self.world_state["player"]
        inventory_list = ", ".join(player["inventory"]) if player["inventory"] else "Empty"
        
        return f"Health: {player['health']}/{player['max_health']} | Gold: {player['gold']} | Inventory: {inventory_list}"

    @function_tool()
    async def add_to_inventory(self, context: RunContext, item: str) -> str:
        """Add an item to player's inventory"""
        self.world_state["player"]["inventory"].append(item)
        return f"Added {item} to inventory. {await self.check_inventory(context)}"

    @function_tool()
    async def remove_from_inventory(self, context: RunContext, item: str) -> str:
        """Remove an item from player's inventory"""
        if item in self.world_state["player"]["inventory"]:
            self.world_state["player"]["inventory"].remove(item)
            return f"Removed {item} from inventory. {await self.check_inventory(context)}"
        return f"{item} not found in inventory."

    @function_tool()
    async def update_health(self, context: RunContext, change: int) -> str:
        """Update player's health (positive or negative)"""
        self.world_state["player"]["health"] = max(0, min(
            self.world_state["player"]["health"] + change,
            self.world_state["player"]["max_health"]
        ))
        
        status = "gained" if change > 0 else "lost"
        current_health = self.world_state["player"]["health"]
        
        if current_health <= 0:
            self.game_active = False
            return f"You have {status} {abs(change)} health. Current health: 0/100. You have been defeated!"
        
        return f"You have {status} {abs(change)} health. Current health: {current_health}/100."

    @function_tool()
    async def move_to_location(self, context: RunContext, location: str) -> str:
        """Move player to a new location"""
        valid_locations = list(self.world_state["locations"].keys())
        if location not in valid_locations:
            return f"Unknown location. Valid locations: {', '.join(valid_locations)}"
        
        self._update_location(location)
        loc_data = self.world_state["locations"][location]
        return f"You have moved to {loc_data['name']}. {loc_data['description']}"

    @function_tool()
    async def complete_quest(self, context: RunContext, quest_name: str) -> str:
        """Mark a quest as completed"""
        if quest_name in self.world_state["quests"]:
            self.world_state["quests"][quest_name]["status"] = "completed"
            self.world_state["player"]["reputation"] += 10
            return f"Quest '{quest_name}' completed! Your reputation has increased."
        return f"Quest '{quest_name}' not found."

    @function_tool()
    async def get_world_state(self, context: RunContext) -> str:
        """Get current world state summary (for GM context)"""
        player = self.world_state["player"]
        current_loc = self.world_state["locations"][player["location"]]
        
        active_quests = [q for q in self.world_state["quests"].values() if q["status"] == "active"]
        quest_status = active_quests[0]["name"] if active_quests else "No active quests"
        
        return f"Location: {current_loc['name']} | Health: {player['health']}/100 | Active Quest: {quest_status} | Turn: {self.world_state['game_state']['turn_count']}"

    @function_tool()
    async def restart_game(self, context: RunContext) -> str:
        """Restart the game with fresh world state"""
        self.world_state = self._initialize_world_state()
        self.game_active = True
        return "Game has been restarted. You find yourself at the beginning of your adventure in Oakhaven Village."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Initialize the game master agent
    agent = GameMasterAgent()
    
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