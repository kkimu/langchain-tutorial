import { tool } from "@langchain/core/tools";
import { z } from "zod";

const addTool = tool(
  async ({ a, b }) => {
    return a + b;
  },
  {
    name: "add",
    schema: z.object({
      a: z.number(),
      b: z.number(),
    }),
    description: "Adds a and b.",
  }
);

const multiplyTool = tool(
  async ({ a, b }) => {
    return a * b;
  },
  {
    name: "multiply",
    schema: z.object({
      a: z.number(),
      b: z.number(),
    }),
    description: "Multiplies a and b.",
  }
);

const tools = [addTool, multiplyTool];

import { HumanMessage } from "@langchain/core/messages";
import { llm } from "./common/model.ts";

const llmWithTools = llm.bindTools(tools);

const messages = [new HumanMessage("What is 3 * 12? Also, what is 11 + 49?")];

const aiMessage = await llmWithTools.invoke(messages);

console.log(aiMessage);

messages.push(aiMessage);

const toolsByName = {
  add: addTool,
  multiply: multiplyTool,
};

for (const toolCall of aiMessage.tool_calls!) {
  const selectedTool = toolsByName[toolCall.name];
  const toolMessage = await selectedTool.invoke(toolCall);
  messages.push(toolMessage);
}

console.log(messages);

const res = await llmWithTools.invoke(messages);

console.log(res);
