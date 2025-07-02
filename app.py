import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import asyncio
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from concurrent.futures import as_completed
from typing import Any, Dict, List, Optional, Tuple
import re

import dotenv
import streamlit as st
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# Correctly import the agent manager class
from src.aoai.aoai_helper import AzureOpenAIManager
from usecases.agenticrag.prompts import (
    SYSTEM_PROMPT_PLANNER,
    SYSTEM_PROMPT_SUMMARY,
    SYSTEM_PROMPT_VERIFIER,
    generate_final_summary,
    generate_user_prompt,
    generate_verifier_prompt,
)
from usecases.agenticrag.settings import (
    AZURE_AI_FOUNDRY_AGENT_IDS,
    AZURE_AI_FOUNDRY_FABRIC_AGENT,
    AZURE_AI_FOUNDRY_SHAREPOINT_AGENT,
    AZURE_AI_FOUNDRY_WEB_AGENT,
    CUSTOM_AGENT_NAMES,
    PLANNER_AGENT,
    SUMMARY_AGENT,
    VERIFIER_AGENT,
)
from usecases.agenticrag.tools import run_agent
from utils.ml_logging import get_logger

# --- Logging ---
logger = get_logger()

# --- Type Aliases ---
AgentStatusDict = Dict[str, str]
AgentResponseDict = Dict[str, Optional[str]]

# --- UI Constants ---
PLANNER = "PlannerAgent"
SP = "SharePointDataRetrievalAgent"
WEB = "BingDataRetrievalAgent"
FAB = "FabricDataRetrievalAgent"
VERIFY = "VerifierAgent"
SUMMARY = "SummaryAgent"

ICONS = {
    PLANNER: "🧩",
    SP: "📖",
    WEB: "🔎",
    FAB: "🛠️",
    VERIFY: "✅",
    SUMMARY: "📝",
}
LABELS = {
    PLANNER: "Planner",
    SP: "SP",
    WEB: "Bing",
    FAB: "Fabric",
    VERIFY: "Verifier",
    SUMMARY: "Summ..",
}

WIDTH, HEIGHT = 300, 420
NODE_W, NODE_H = 96, 36
COL_LEFT = 20
COL_CENTER = WIDTH // 2 - NODE_W // 2
COL_RIGHT = WIDTH - NODE_W - 20

NODE_POS = {
    PLANNER: (COL_CENTER, 24),
    SP: (COL_LEFT, 120),
    WEB: (COL_CENTER, 120),
    FAB: (COL_RIGHT, 120),
    VERIFY: (COL_CENTER, 230),
    SUMMARY: (COL_CENTER, 335),
}

EDGES = [
    (PLANNER, SP),
    (PLANNER, WEB),
    (PLANNER, FAB),
    (SP, VERIFY),
    (WEB, VERIFY),
    (FAB, VERIFY),
    (VERIFY, SUMMARY),
]

STATUS_COLOURS = {
    "pending": ("#f5f5f5", "#9e9e9e", ""),
    "running": ("#fff7d1", "#e6b800", "0 0 16px 4px #ffe066"),
    "done": ("#e8f5e9", "#00b050", "0 0 10px 2px #b2f2bb"),
    "error": ("#ffebee", "#d32f2f", "0 0 16px 4px #ff6b6b"),
    "denied": ("#ffebee", "#d32f2f", "0 0 16px 4px #ff6b6b"),
    "approved": ("#e8f5e9", "#00b050", "0 0 16px 4px #69f0ae"),
}

def render_agent_mind_map(agent_status):
    # Placeholder: implement visualization if needed
    pass

def setup_environment() -> None:
    """Initialize environment variables and session state."""
    logger.info("Setting up environment and initialising session state.")
    dotenv.load_dotenv(".env", override=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "project_client" not in st.session_state:
        logger.info("Initialising Azure AI Project Client.")
        # Correctly initialize MLClient using the connection string and credential
        st.session_state.project_client = MLClient(
            credential=DefaultAzureCredential(),
            connection_string=os.environ["AZURE_AI_FOUNDRY_CONNECTION_STRING"]
        )

    for agent_key, config_path in CUSTOM_AGENT_NAMES.items():
        if agent_key not in st.session_state:
            logger.info(f"Loading agent '{agent_key}' from config: {config_path}")
            # Correctly instantiate the agent manager
            st.session_state[agent_key] = AzureOpenAIManager()


def render_chat_history(chat_container: Any) -> None:
    """Render the chat history in the Streamlit container."""
    logger.debug("Rendering chat history.")
    for msg in st.session_state.chat_history:
        role = msg["role"].lower()
        content = msg["content"]
        avatar = msg.get("avatar", "🤖")

        if role in ("info", "system"):
            st.info(content, icon=avatar)
        elif role == "user":
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(content, unsafe_allow_html=True)
        elif role == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(content, unsafe_allow_html=True)
        else:  # agent responses
            with st.expander(f"{avatar} {msg['role']} says...", expanded=False):
                st.markdown(content, unsafe_allow_html=True)


def select_agents(current_query: str) -> Optional[Dict[str, Any]]:
    """Select which agents are needed for the query."""
    logger.info(f"Selecting agents for query: {current_query}")
    try:
        st.session_state.agent_status[PLANNER] = "running"
        render_agent_mind_map(st.session_state.agent_status)

        # Unpack the response robustly in case it's a tuple
        result = asyncio.run(
            st.session_state[PLANNER_AGENT].async_generate_chat_completion_response(
                conversation_history=[],
                query=generate_user_prompt(current_query),
                system_message_content=SYSTEM_PROMPT_PLANNER,
            )
        )
        logger.debug(f"Planner agent raw result: {result} (type: {type(result)})")
        print("Planner agent raw result:", result, "type:", type(result))
        response = result
        # Unpack recursively if result is a tuple
        while isinstance(response, tuple):
            logger.debug(f"Unpacking tuple: {response} (type: {type(response)})")
            response = response[0]
        logger.debug(f"Planner agent response after unpacking: {response} (type: {type(response)})")

        import re
        agents = {}
        content = None
        # If response is a ChatCompletion object, extract the content
        if hasattr(response, "choices") and response.choices:
            content = response.choices[0].message.content
        elif isinstance(response, dict) and "response" in response:
            inner = response["response"]
            if isinstance(inner, str):
                content = inner
            elif isinstance(inner, dict):
                agents = inner
        elif isinstance(response, dict):
            agents = response

        # If content is a string, clean and parse it as JSON
        if content and isinstance(content, str):
            content = content.strip()
            if content.startswith("```"):
                content = content.split('\n', 1)[-1]
                if content.endswith("```"):
                    content = content.rsplit("```", 1)[0]
                content = content.strip()
            if content.lower().startswith("json"):
                content = content[4:].strip()
            match = re.search(r'{.*}', content, re.DOTALL)
            if match:
                content = match.group(0)
            try:
                agents = json.loads(content)
            except Exception as e:
                logger.error(f"Failed to parse planner agent response as JSON: {e}")
                st.error(f"Planner agent returned invalid JSON: {content}")
                agents = {}
        # --- NEW: Wrap if needed ---
        if isinstance(agents, dict) and "agents_needed" in agents:
            agents = {"response": agents}

        # Fallback: if no agents selected and query is generic/conversational, select Bing
        if not agents or not agents.get("response") or not agents["response"].get("agents_needed"):
            generic_words = ["hello", "hi", "how are you", "what's up", "hey", "greetings", "good morning", "good afternoon", "good evening"]
            if any(word in current_query.lower() for word in generic_words):
                agents = {
                    "response": {
                        "agents_needed": ["BingDataRetrievalAgent"],
                        "justification": "The query is a greeting or small talk, so BingDataRetrievalAgent is best suited to provide a helpful response."
                    }
                }
            else:
                st.warning("No agents selected. Please refine your query.")
                return None

        logger.debug(f"Planner agent response: {agents}")

        st.session_state.agent_status[PLANNER] = "done"
        render_agent_mind_map(st.session_state.agent_status)

        if not agents or not agents.get("response") or not agents["response"].get("agents_needed"):
            st.warning("No agents selected. Please refine your query.")
            return None
        
        st.info(
            f"**Agents Selected:** {', '.join(agents['response']['agents_needed'])}\n"
            f"**Justification:** {agents['response']['justification']}",
            icon="ℹ️",
        )
        return agents

    except Exception as e:
        st.error(f"Planner agent selection failed: {e}")
        logger.exception("Planner agent selection failed")
        return None


def run_selected_agents(
    agents_needed: List[str], current_query: str
) -> AgentResponseDict:
    """Run selected retriever agents in parallel and return their responses."""
    logger.info(f"Running agents in parallel: {agents_needed}")
    dicta: AgentResponseDict = {}
    results: List[Tuple[str, Optional[str], Optional[str]]] = []

    local_status: AgentStatusDict = {a: "pending" for a in agents_needed}
    render_agent_mind_map({**st.session_state.agent_status, **local_status})

    def worker(agent_id: str, query: str, pc: MLClient, name: str):
        try:
            return "running", None, run_agent(pc, agent_id, query)
        except Exception as exc:
            logger.error(f"{name} failed: {exc}")
            return "error", str(exc), (None, None)

    with ThreadPoolExecutor(max_workers=len(agents_needed)) as executor:
        future_map = {}
        for ag in agents_needed:
            if ag not in AZURE_AI_FOUNDRY_AGENT_IDS:
                logger.error(f"{ag} missing in AZURE_AI_FOUNDRY_AGENT_IDS.")
                results.append((ag, None, "Not configured"))
                local_status[ag] = "error"
                continue

            fid = AZURE_AI_FOUNDRY_AGENT_IDS[ag]
            fut = executor.submit(
                worker, fid, current_query, st.session_state.project_client, ag
            )
            future_map[fut] = ag

        for fut in as_completed(future_map):
            ag = future_map[fut]
            status_flag, err, (resp, _) = fut.result()
            local_status[ag] = (
                "done" if status_flag == "running" and resp else status_flag
            )
            render_agent_mind_map({**st.session_state.agent_status, **local_status})
            results.append((ag, resp, err))

    for ag in agents_needed:
        item = next((x for x in results if x[0] == ag), None)
        if not item:
            continue
        _, resp, err = item
        avatar = ICONS.get(ag, "🤖")
        with st.expander(f"{avatar} {ag} says...", expanded=False):
            if err or resp is None:
                st.warning(f"Agent {ag} failed: {err or 'No response.'}")
            else:
                st.markdown(resp, unsafe_allow_html=True)

        st.session_state.chat_history.append(
            {
                "role": ag,
                "content": err or resp or "No response",
                "avatar": avatar,
                "error": bool(err or resp is None),
            }
        )
        if resp:
            dicta[ag] = resp

    for k, v in local_status.items():
        st.session_state.agent_status[k] = v

    logger.info(f"Collected retriever responses: {list(dicta.keys())}")
    return dicta


def evaluate_with_verifier(
    current_query: str, dicta: AgentResponseDict
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Run verifier agent and return its decision."""
    logger.info("Running verifier agent")
    st.session_state.agent_status[VERIFY] = "running"
    render_agent_mind_map(st.session_state.agent_status)

    try:
        evaluation = asyncio.run(
            st.session_state[VERIFIER_AGENT].async_generate_chat_completion_response(
                conversation_history=[],
                query=generate_verifier_prompt(
                    current_query,
                    fabric_data_summary=dicta.get(AZURE_AI_FOUNDRY_FABRIC_AGENT),
                    sharepoint_data_summary=dicta.get(
                        AZURE_AI_FOUNDRY_SHAREPOINT_AGENT
                    ),
                    bing_data_summary=dicta.get(AZURE_AI_FOUNDRY_WEB_AGENT),
                ),
                system_message_content=SYSTEM_PROMPT_VERIFIER,
                max_tokens=400,
            )
        )
    except Exception as exc:
        logger.exception("Verifier agent crashed")
        st.session_state.agent_status[VERIFY] = "error"
        render_agent_mind_map(st.session_state.agent_status)
        st.error(f"Verifier agent exception: {exc}")
        return None, None, None

    logger.debug(f"Verifier raw: {evaluation}")
    response_obj = None
    content = None

    # If evaluation is a ChatCompletion object, extract the content
    if hasattr(evaluation, "choices") and evaluation.choices:
        content = evaluation.choices[0].message.content
    elif isinstance(evaluation, dict) and "response" in evaluation:
        inner = evaluation["response"]
        if isinstance(inner, str):
            content = inner
        elif isinstance(inner, dict):
            response_obj = inner
    elif isinstance(evaluation, dict):
        response_obj = evaluation

    # If content is a string, clean and parse it as JSON
    if content and isinstance(content, str):
        content = content.strip()
        if content.startswith("```"):
            content = content.split('\n', 1)[-1]
            if content.endswith("```"):
                content = content.rsplit("```", 1)[0]
            content = content.strip()
        if content.lower().startswith("json"):
            content = content[4:].strip()
        match = re.search(r'{.*}', content, re.DOTALL)
        if match:
            content = match.group(0)
        try:
            response_obj = json.loads(content)
        except Exception as e:
            logger.error(f"Failed to parse verifier response as JSON: {e}")
            st.error(f"Verifier returned invalid JSON: {content}")
            response_obj = None

    if not response_obj:
        st.error(f"Verifier returned unexpected format: {evaluation}")
        return None, None, None

    status = response_obj.get("status")
    resp_txt = response_obj.get("response", "")
    rewritten_query = response_obj.get("rewritten_query", "")

    st.session_state.agent_status[VERIFY] = (
        "approved" if status == "Approved" else "denied"
    )
    render_agent_mind_map(st.session_state.agent_status)

    avatar = "✅" if status == "Approved" else "❌"
    with st.expander(f"{avatar} {VERIFY} says...", expanded=False):
        st.markdown(
            f"**{status}:** {resp_txt if status == 'Approved' else rewritten_query}",
            unsafe_allow_html=True,
        )

    st.session_state.chat_history.append(
        {
            "role": VERIFY,
            "content": resp_txt if status == "Approved" else rewritten_query,
            "avatar": avatar,
        }
    )
    return status, resp_txt, rewritten_query


def summarize_results(
    initial_message: str, dicta: AgentResponseDict, chat_container: Any
) -> None:
    """Summarize results and reply as assistant."""
    logger.info("Running summary agent")
    st.session_state.agent_status[SUMMARY] = "running"
    render_agent_mind_map(st.session_state.agent_status)

    try:
        summary_content = asyncio.run(
            st.session_state[SUMMARY_AGENT].run(
                user_prompt=generate_final_summary(initial_message, dicta=dicta),
                conversation_history=[],
                system_message_content=SYSTEM_PROMPT_SUMMARY,
                max_tokens=3000,
            )
        )
        logger.debug(f"Summary raw: {summary_content}")

        summary_resp = None
        if isinstance(summary_content, dict) and "response" in summary_content:
            inner = summary_content["response"]
            summary_resp = inner if isinstance(inner, str) else inner.get("response", "")

        if not summary_resp:
            st.session_state.agent_status[SUMMARY] = "error"
            render_agent_mind_map(st.session_state.agent_status)
            with chat_container:
                st.error("Summary agent failed to produce a response.")
            return

        st.session_state.agent_status[SUMMARY] = "done"
        render_agent_mind_map(st.session_state.agent_status)

        with chat_container:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(summary_resp, unsafe_allow_html=True)
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": summary_resp,
                "avatar": "🤖",
            }
        )
        st.toast("📧 An email with the results of your query has been sent!", icon="📩")
    except Exception as e:
        logger.error(f"Summary agent failed: {e}")
        st.error(f"Summary agent failed: {e}")
        st.session_state.agent_status[SUMMARY] = "error"
        render_agent_mind_map(st.session_state.agent_status)


def main() -> None:
    """Main entry point for the Streamlit app."""
    try:
        st.set_page_config(page_title="R+D Intelligent Multi-Agent Assistant", layout="wide")
        setup_environment()

        st.markdown(
            """
            <style>
            .titleContainer {
                text-align: center;
                background: linear-gradient(145deg, #1F6095, #008AD7);
                color: #FFFFFF;
                padding: 35px;
                border-radius: 12px;
                margin-bottom: 25px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
            }
            .titleContainer h1 {
                margin: 0;
                font-size: 2rem;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                font-weight: 600;
                letter-spacing: 0.8px;
            }
            .titleContainer h3 {
                margin: 8px 0 0;
                font-size: 1rem;
                font-weight: 400;
            }
            </style>
            <div class="titleContainer">
                <h1>R+D Intelligent Assistant 🤖</h1>
                <h3>powered by Azure AI Foundry Agent Service</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        agents_for_map = [PLANNER, SP, WEB, FAB, VERIFY, SUMMARY]
        if "agent_status" not in st.session_state:
            st.session_state.agent_status = {a: "pending" for a in agents_for_map}

        render_agent_mind_map(st.session_state.agent_status)

        user_input = st.chat_input("Ask your R+D query here...")
        chat_container = st.container(height=500)

        with chat_container:
            render_chat_history(chat_container)

            if user_input:
                st.session_state.chat_history.append(
                    {"role": "user", "content": user_input, "avatar": "🧑‍💻"}
                )
                with st.chat_message("user", avatar="🧑‍💻"):
                    st.markdown(user_input, unsafe_allow_html=True)

                initial_message = user_input
                current_query = user_input
                MAX_RETRIES = 3

                for attempt in range(1, MAX_RETRIES + 1):
                    with st.spinner(f"Agents collaborating... Attempt {attempt}"):
                        try:
                            agents_dict = select_agents(current_query)
                            if not agents_dict:
                                break

                            selected_agents = agents_dict["response"]["agents_needed"]
                            dicta = run_selected_agents(selected_agents, current_query)

                            status, _, rewritten = evaluate_with_verifier(
                                current_query, dicta
                            )

                            if status == "Approved":
                                summarize_results(
                                    initial_message, dicta, chat_container
                                )
                                break
                            elif status == "Denied" and rewritten:
                                current_query = rewritten
                                st.info(
                                    f"Verifier requested retry with rewritten query:\n\n'{rewritten}'",
                                    icon="ℹ️",
                                )
                            else:
                                st.warning(
                                    "Verifier denied but no rewritten query provided. Stopping."
                                )
                                break
                        except Exception as e:
                            logger.error(f"Error in agent workflow: {e}")
                            st.error(f"An error occurred in the agent workflow: {e}")
                            break
                else:
                    st.warning("Maximum retries reached. Please refine your query.")

    except Exception as e:
        logger.error(f"A critical error occurred in the application: {e}", exc_info=True)
        st.error(f"A critical error occurred: {e}")

if __name__ == "__main__":
    main()