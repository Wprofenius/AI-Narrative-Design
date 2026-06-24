import os
import time
import logging
from openai import OpenAI, APIConnectionError, RateLimitError, APIError

# ==========================================
# Configuration & Setup
# ==========================================
# Configure logging to replace standard print() statements. 
# This is an industry standard for debugging production applications.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize client. Ensure OPENAI_API_KEY is set in your environment variables.
try:
    client = OpenAI()
except Exception as e:
    logging.error(f"Failed to initialize OpenAI client. Check your environment variables. Error: {e}")
    raise

# ==========================================
# Main Generator Function
# ==========================================
def generate_npc_dialogue(
    player_action: str, 
    npc_role: str, 
    emotion: str, 
    environment: str, 
    standing: str, 
    model: str = "gpt-4o",
    max_retries: int = 3
) -> str:
    """
    Generates deeply contextual, era-appropriate dialogue with built-in
    error handling, input validation, and automatic retry logic.
    """
    
    # 1. Input Validation: Prevent sending empty requests to the API
    if not player_action.strip():
        logging.warning("Player action is empty. Returning default silence.")
        return "... (The character stares in silence.)"

    # 2. System Prompt Architecture
    system_instruction = f"""You are the lead narrative designer and dialogue generator for 'Broken Oath: Grunwald', set in 15th-century Prussia.
    
    Current Character Profile:
    - Role: {npc_role}
    - Current Emotional State: {emotion}
    - Faction Standing with Player: {standing}
    
    Scene Atmosphere:
    - {environment}
    
    Core Directives:
    - Tone: Reflect the character's emotional state and standing with the player, while maintaining medieval authenticity.
    - Environmental Storytelling: Passively weave the weather or atmosphere into the dialogue if natural.
    - Vocabulary: Use era-appropriate terminology (circa 1400s). Strictly avoid modern phrasing.
    - Formatting: Output only the spoken dialogue in quotes, followed by a brief parenthetical action description. 
    """

    # 3. Execution & Retry Logic
    for attempt in range(max_retries):
        try:
            logging.info(f"Attempt {attempt + 1}: Generating dialogue for {npc_role}...")
            
            response = client.chat.completions.create(
                model=model,
                temperature=0.7, 
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": f"The player character has just: {player_action}"}
                ]
            )
            
            output = response.choices[0].message.content
            
            # Output Validation: Ensure the AI didn't return an empty string
            if not output:
                raise ValueError("Model returned an empty response.")
                
            return output

        # Catch specific OpenAI errors to handle them appropriately
        except RateLimitError:
            wait_time = (2 ** attempt)  # Exponential backoff: 1s, 2s, 4s...
            logging.warning(f"Rate limit hit. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            
        except APIConnectionError:
            logging.error("Failed to connect to the OpenAI API. Check your internet connection.")
            break # Exit loop, no point retrying if offline
            
        except (APIError, ValueError) as e:
            logging.error(f"API or Validation Error: {e}")
            break # Exit loop for structural errors

        except Exception as e:
            logging.critical(f"An unexpected error occurred: {e}")
            break

    # 4. Fallback Mechanism (Failsafe for the game engine)
    logging.error("All generation attempts failed. Returning fallback dialogue.")
    return '"I have nothing to say to you right now." (They turn away abruptly.)'

# ==========================================
# Execution Block (Testing the Workflow)
# ==========================================
if __name__ == "__main__":
    
    # Scenario 1: A complex, high-tension interaction
    action_1 = "Approached the campfire, hand resting casually on his sword hilt."
    
    output_1 = generate_npc_dialogue(
        player_action=action_1,
        npc_role="Battle-weary Teutonic Knight",
        emotion="suspicious and cold",
        environment="Freezing rain turning the mud to slush.",
        standing="Distrusted (outsider)"
    )
    
    print("\n--- Final Game Engine Output ---")
    print(f"Player Action: {action_1}")
    print(f"Result: {output_1}\n")
