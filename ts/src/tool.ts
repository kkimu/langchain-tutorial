import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { llm } from "./common/model.ts";
/**
 * Note that the descriptions here are crucial, as they will be passed along
 * to the model along with the class name.
 */
const calculatorSchema = z.object({
  operation: z
    .enum(["add", "subtract", "multiply", "divide"])
    .describe("The type of operation to execute."),
  number1: z.number().describe("The first number to operate on."),
  number2: z.number().describe("The second number to operate on."),
});

const calculatorTool = tool(
  async ({ operation, number1, number2 }) => {
    // Functions must return strings
    if (operation === "add") {
      return `${number1 + number2}`;
    } else if (operation === "subtract") {
      return `${number1 - number2}`;
    } else if (operation === "multiply") {
      return `${number1 * number2}`;
    } else if (operation === "divide") {
      return `${number1 / number2}`;
    } else {
      throw new Error("Invalid operation.");
    }
  },
  {
    name: "calculator",
    description: "Can perform mathematical operations.",
    schema: calculatorSchema,
  }
);

const llmWithTools = llm.bindTools([calculatorTool]);

const res = await llmWithTools.invoke("What is 3 * 12? Also, what is 11 + 49?");

console.log(res);
