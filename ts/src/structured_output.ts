import { z } from "zod";
import { model } from "./common/model.ts";

const joke = z.object({
  setup: z.string().describe("The setup of the joke"),
  punchline: z.string().describe("The punchline to the joke"),
  rating: z.number().optional().describe("How funny the joke is, from 1 to 10"),
});

const structuredLlm = model.withStructuredOutput(joke);

await structuredLlm.invoke("Tell me a joke about cats");
