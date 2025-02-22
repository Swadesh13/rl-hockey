from __future__ import annotations

import argparse

from comprl.client import Agent, launch_client
from henv.hockey_agent import HockeyCompetetionAgent
from utils.load import LoadTD7Agents
from ppo.load_ppo_models import load_all_ppo_agents,load_all_sac_agents

# Function to initialize the agent.  This function is used with `launch_client` below,
# to lauch the client and connect to the server.
def initialize_agent(agentArgs: list[str]) -> Agent:
    return HockeyCompetetionAgent(agent=LoadTD7Agents()["td7_all_big"])

def main() -> None:
    launch_client(initialize_agent)


if __name__ == "__main__":
    main()
