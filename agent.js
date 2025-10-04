import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { MessagesAnnotation, StateGraph } from "@langchain/langgraph";
import { ToolMessage } from "@langchain/core/messages";
import { config } from "dotenv";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import readline from "readline";

config();

// -------------------------
// Define LLM
// -------------------------
const llm = new ChatGoogleGenerativeAI({
  apiKey: process.env.GEMINI_API_KEY,
  model: "gemini-2.0-flash",
});

// -------------------------
// Define Tools
// -------------------------
const tools = [
  tool(async ({ a, b }) => `${a * b}`, { name: "multiply", description: "Multiply two numbers", schema: z.object({ a: z.number(), b: z.number() }) }),
  tool(async ({ a, b }) => `${a + b}`, { name: "add", description: "Add two numbers", schema: z.object({ a: z.number(), b: z.number() }) }),
  tool(async ({ a, b }) => `${a - b}`, { name: "subtract", description: "Subtract two numbers", schema: z.object({ a: z.number(), b: z.number() }) }),
  tool(async ({ a, b }) => `${a / b}`, { name: "divide", description: "Divide two numbers", schema: z.object({ a: z.number(), b: z.number() }) }),
  tool(async ({ a, b }) => `${a % b}`, { name: "mod", description: "Remainder of a/b", schema: z.object({ a: z.number(), b: z.number() }) }),
  tool(async ({ a, b }) => `${a ** b}`, { name: "power", description: "a to the power b", schema: z.object({ a: z.number(), b: z.number() }) }),
  tool(async ({ a }) => `${Math.sqrt(a)}`, { name: "sqrt", description: "Square root of a", schema: z.object({ a: z.number() }) }),
  tool(async ({ a }) => `${Math.abs(a)}`, { name: "abs", description: "Absolute value", schema: z.object({ a: z.number() }) }),
  tool(async ({ a }) => `${Math.sin(a)}`, { name: "sin", description: "Sine (radians)", schema: z.object({ a: z.number() }) }),
  tool(async ({ a }) => `${Math.cos(a)}`, { name: "cos", description: "Cosine (radians)", schema: z.object({ a: z.number() }) }),
  tool(async ({ a }) => `${Math.tan(a)}`, { name: "tan", description: "Tangent (radians)", schema: z.object({ a: z.number() }) }),
  tool(async ({ a }) => `${Math.asin(a)}`, { name: "arcsin", description: "Arcsine (radians)", schema: z.object({ a: z.number() }) }),
  tool(async ({ a }) => `${Math.acos(a)}`, { name: "arccos", description: "Arccosine (radians)", schema: z.object({ a: z.number() }) }),
  tool(async ({ a }) => `${Math.atan(a)}`, { name: "arctan", description: "Arctangent (radians)", schema: z.object({ a: z.number() }) }),
  tool(async ({ a }) => `${Math.log(a)}`, { name: "log", description: "Natural log", schema: z.object({ a: z.number() }) }),
  tool(async ({ a }) => `${Math.log10(a)}`, { name: "log10", description: "Log base 10", schema: z.object({ a: z.number() }) }),
  tool(async ({ a }) => `${Math.exp(a)}`, { name: "exp", description: "e^a", schema: z.object({ a: z.number() }) }),
  tool(async ({ a }) => `${Math.round(a)}`, { name: "round", description: "Round to nearest", schema: z.object({ a: z.number() }) }),
  tool(async ({ a }) => `${Math.floor(a)}`, { name: "floor", description: "Round down", schema: z.object({ a: z.number() }) }),
  tool(async ({ a }) => `${Math.ceil(a)}`, { name: "ceil", description: "Round up", schema: z.object({ a: z.number() }) }),
  tool(async ({ a, b }) => `${Math.max(a, b)}`, { name: "max", description: "Max of two numbers", schema: z.object({ a: z.number(), b: z.number() }) }),
  tool(async ({ a, b }) => `${Math.min(a, b)}`, { name: "min", description: "Min of two numbers", schema: z.object({ a: z.number(), b: z.number() }) }),
];

const toolsByName = Object.fromEntries(tools.map((tool) => [tool.name, tool]));
const llmWithTools = llm.bindTools(tools);

// -------------------------
// Graph Nodes
// -------------------------
async function llmCall(state) {
  const result = await llmWithTools.invoke(
    [
      { role: "system", content: "You are a helpful assistant performing arithmetic tasks." },
      ...state.messages,
    ],
    {
      configurable: {
        run_name: "LLMCall",
        tags: ["gemini", "arithmetic", "langsmith"],
      },
    }
  );
  return { messages: [result] };
}

async function ToolNode(state) {
  const results = [];
  const lastMessage = state.messages.at(-1);
  if (lastMessage?.tool_calls?.length) {
    for (const toolCall of lastMessage.tool_calls) {
      const tool = toolsByName[toolCall.name];
      const observation = await tool.invoke(toolCall.args, {
        configurable: { run_name: `Tool-${toolCall.name}`, tags: ["tool", "langsmith"] },
      });
      results.push(new ToolMessage({ content: observation, tool_call_id: toolCall.id }));
    }
  }
  return { messages: results };
}

function shouldContinue(state) {
  const lastMessage = state.messages.at(-1);
  return lastMessage?.tool_calls?.length ? "Action" : "__end__";
}

// -------------------------
// Build Agent
// -------------------------
const agentBuilder = new StateGraph(MessagesAnnotation)
  .addNode("llmCall", llmCall)
  .addNode("tools", ToolNode)
  .addEdge("__start__", "llmCall")
  .addConditionalEdges("llmCall", shouldContinue, { Action: "tools", __end__: "__end__" })
  .addEdge("tools", "llmCall")
  .compile();

// -------------------------
// Console Input
// -------------------------
const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

rl.question("Enter your query: ", async (userInput) => {
  const messages = [{ role: "user", content: userInput }];

  const result = await agentBuilder.invoke(
    { messages },
    { configurable: { run_name: "ArithmeticAgentRun", tags: ["console", "gemini", "langsmith"] } }
  );

  console.log("\n--- Conversation Trace ---");
  for (const msg of result.messages) {
    if (msg._getType() === "human") console.log(`User: ${msg.content}`);
    else if (msg._getType() === "ai") {
      if (msg.tool_calls?.length) {
        for (const call of msg.tool_calls) {
          console.log(`AI decided: Call tool "${call.name}" with args ${JSON.stringify(call.args)}`);
        }
      } else console.log(`AI: ${msg.content}`);
    } else if (msg._getType() === "tool") console.log(`Tool Result: ${msg.content}`);
  }

  console.log("\n--- Final Answer ---");
  const lastAI = result.messages.filter(m => m._getType() === "ai").at(-1);
  console.log(lastAI.content);

  rl.close();
});
