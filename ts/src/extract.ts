import { ChatPromptTemplate } from "@langchain/core/prompts";
import { z } from "zod";
import { model } from "./common/model.ts";

// Define a custom prompt to provide instructions and any additional context.
// 1) You can add examples into the prompt template to improve extraction quality
// 2) Introduce additional parameters to take context into account (e.g., include metadata
//    about the document from which the text was extracted.)
const promptTemplate = ChatPromptTemplate.fromMessages([
  [
    "system",
    `You are an expert extraction algorithm.
  Only extract relevant information from the text.
  If you do not know the value of an attribute asked to extract,
  return null for the attribute's value.`,
  ],
  // Please see the how-to about improving performance with
  // reference examples.
  // ["placeholder", "{examples}"],
  ["human", "{text}"],
]);

const person = z.object({
  name: z.optional(z.string()).describe("The name of the person"),
  hair_color: z
    .optional(z.string())
    .describe("The color of the person's hair if known"),
  height_in_meters: z.number().nullish().describe("Height measured in meters"),
});

const dataSchema = z.object({
  people: z.array(person).describe("Extracted data about people"),
});

const structured_model = model.withStructuredOutput(dataSchema);
const prompt3 = await promptTemplate.invoke({
  text: "My name is Jeff, my hair is black and i am 6 feet tall. Anna has the same color hair as me.",
});

const res = await structured_model.invoke(prompt3);

console.log(res);
