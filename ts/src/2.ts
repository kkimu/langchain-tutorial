import { ChatAnthropic } from "@langchain/anthropic";
import { ChatPromptTemplate } from "@langchain/core/prompts";

const model = new ChatAnthropic({
  model: "claude-3-5-sonnet-20240620",
  temperature: 0,
});

const systemTemplate = "Translate the following from English into {language}";

const promptTemplate = ChatPromptTemplate.fromMessages([
  ["system", systemTemplate],
  ["user", "{text}"],
]);

const promptValue = await promptTemplate.invoke({
  language: "italian",
  text: "hi!",
});

console.log(promptValue);

const response = await model.invoke(promptValue);
console.log(`${response.content}`);
