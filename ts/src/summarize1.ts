import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { llm } from "./common/model.ts";

const pTagSelector = "p";
const cheerioLoader = new CheerioWebBaseLoader(
  "https://lilianweng.github.io/posts/2023-06-23-agent/",
  {
    selector: pTagSelector,
  }
);

const docs = await cheerioLoader.load();

// Define prompt
const prompt = PromptTemplate.fromTemplate(
  "Summarize the main themes in these retrieved docs: {context}"
);

// Instantiate
const chain = await createStuffDocumentsChain({
  llm: llm,
  outputParser: new StringOutputParser(),
  prompt,
});

// // Invoke
// const result = await chain.invoke({ context: docs });
// console.log(result);

const stream = await chain.stream({ context: docs });

for await (const token of stream) {
  process.stdout.write(token + "|");
}
