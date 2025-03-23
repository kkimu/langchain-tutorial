import { ChatAnthropic } from "@langchain/anthropic";

// export const model = new ChatAnthropic({
//   model: "claude-3-7-sonnet-20250219",
//   temperature: 0,
// });

export const model = new ChatAnthropic({
  model: "claude-3-5-haiku-20241022",
  temperature: 0,
});

export const llm = model;
