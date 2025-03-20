import { ChatAnthropic } from "@langchain/anthropic";

export const model = new ChatAnthropic({
  model: "claude-3-5-sonnet-20240620",
  temperature: 0,
});

export const llm = model;
