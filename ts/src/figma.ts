import { FigmaFileLoader } from "@langchain/community/document_loaders/web/figma";

const loader = new FigmaFileLoader({
  nodeIds: [process.env.FIGMA_NODE_ID!],
  fileKey: process.env.FIGMA_FILE_KEY!,
});
const docs = await loader.load();

console.dir(JSON.parse(docs[0].pageContent), { depth: 10 });

// const promptTemplate = ChatPromptTemplate.fromMessages([
//   ["system", "figma frameがどんな画面が説明して"],
//   ["user", "{data}"],
// ]);

// const result = await llm.invoke(
//   await promptTemplate.invoke({ data: docs[0].pageContent })
// );

// console.log(result);
