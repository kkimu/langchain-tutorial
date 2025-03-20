import { ChatAnthropic } from "@langchain/anthropic";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

const model = new ChatAnthropic({
  model: "claude-3-5-sonnet-20240620",
  temperature: 0,
});

const messages = [
  new SystemMessage("Translate the following from English into Italian"),
  new HumanMessage("hi!"),
];

const stream = await model.stream(messages);

const chunks = [];
for await (const chunk of stream) {
  chunks.push(chunk);
  console.log(`${chunk.content}|`);
}
