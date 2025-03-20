import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { HumanMessage } from "@langchain/core/messages";
import { MemorySaver } from "@langchain/langgraph";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { llm } from "./common/model.ts";

// Define the tools for the agent to use
const agentTools = [new TavilySearchResults({ maxResults: 3 })];

// Initialize memory to persist state between graph runs
const agentCheckpointer = new MemorySaver();
const agent = createReactAgent({
  llm: llm,
  tools: agentTools,
  checkpointSaver: agentCheckpointer,
});

// Now it's time to use!
const agentFinalState = await agent.invoke(
  { messages: [new HumanMessage("what is the current weather in sf")] },
  { configurable: { thread_id: "42" } }
);

console.log(
  agentFinalState.messages[agentFinalState.messages.length - 1].content
);

const agentNextState = await agent.invoke(
  { messages: [new HumanMessage("what about ny")] },
  { configurable: { thread_id: "42" } }
);

console.log(
  agentNextState.messages[agentNextState.messages.length - 1].content
);
