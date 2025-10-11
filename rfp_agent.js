import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { MessagesAnnotation, StateGraph } from "@langchain/langgraph";
import { ToolMessage } from "@langchain/core/messages";
import { config } from "dotenv";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import readline from "readline";

config();

// -------------------------
// LLM Setup
// -------------------------
const llm = new ChatGoogleGenerativeAI({
  apiKey: process.env.GEMINI_API_KEY,
  model: "gemini-2.0-flash",
});

// -------------------------
// Define Sub-Agent Tools
// -------------------------
const agentTools = [
  tool(async () => "SalesAgent: identified and summarized one RFP.", {
    name: "SalesAgent",
    description: "Identify RFPs and summarize them.",
    schema: z.object({}),
  }),
  tool(async () => "TechnicalAgent: recommended OEM SKUs with spec match metrics.", {
    name: "TechnicalAgent",
    description: "Match products from RFP with OEM SKUs.",
    schema: z.object({}),
  }),
  tool(async () => "PricingAgent: assigned costs for products and tests.", {
    name: "PricingAgent",
    description: "Assign product and test prices.",
    schema: z.object({}),
  }),
];

const toolsByName = Object.fromEntries(agentTools.map((t) => [t.name, t]));
const llmWithTools = llm.bindTools(agentTools);

// -------------------------
// Node: Master Agent
// -------------------------
async function MasterAgent(state) {
  const result = await llmWithTools.invoke(
    [
      {
        role: "system",
        content: `
You are the Master Agent. Decide freely which sub-agent (SalesAgent, TechnicalAgent, PricingAgent) 
should be called based on the user's query. Only call an agent if needed.
If RFP is found, either as a text snippet or a URL, call the required agent based on query. 
Once you have enough information, provide a consolidated RFP response.
        `,
      },
      ...state.messages,
    ],
    {
      configurable: { run_name: "MasterAgentLLM", tags: ["rfp", "orchestrator", "langsmith"] },
    }
  );
  return { messages: [result] };
}

// -------------------------
// Node: Tool Executor
// -------------------------
async function ToolNode(state) {
  const results = [];
  const lastMessage = state.messages.at(-1);

  if (lastMessage?.tool_calls?.length) {
    for (const call of lastMessage.tool_calls) {
      const tool = toolsByName[call.name];
      if (!tool) continue;

      const observation = await tool.invoke(call.args, {
        configurable: { run_name: `Tool-${call.name}`, tags: ["sub-agent", "rfp"] },
      });

      results.push(
        new ToolMessage({ content: observation, tool_call_id: call.id })
      );
    }
  }

  return { messages: results };
}

// -------------------------
// Flow Control Logic
// -------------------------
function shouldContinue(state) {
  const lastMessage = state.messages.at(-1);
  return lastMessage?.tool_calls?.length ? "Action" : "__end__";
}

// -------------------------
// Build Graph
// -------------------------
const RFPGraph = new StateGraph(MessagesAnnotation)
  .addNode("MasterAgent", MasterAgent)
  .addNode("ToolNode", ToolNode)
  .addEdge("__start__", "MasterAgent")
  .addConditionalEdges("MasterAgent", shouldContinue, { Action: "ToolNode", __end__: "__end__" })
  .addEdge("ToolNode", "MasterAgent")
  .compile();

// -------------------------
// Console Interface
// -------------------------
const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

rl.question("Enter your RFP query: ", async (userInput) => {
  const messages = [{ role: "user", content: userInput }];

  const result = await RFPGraph.invoke(
    { messages },
    { configurable: { run_name: "RFPOrchestratorRun", tags: ["rfp", "multiagent", "langsmith"] } }
  );

  console.log("\n--- Conversation Trace ---");
  for (const msg of result.messages) {
    if (msg._getType() === "human") console.log(`User: ${msg.content}`);
    else if (msg._getType() === "ai") {
      if (msg.tool_calls?.length) {
        for (const call of msg.tool_calls) {
          console.log(`MasterAgent decided: Call "${call.name}"`);
        }
      } else console.log(`MasterAgent: ${msg.content}`);
    } else if (msg._getType() === "tool") {
      console.log(`Agent Response: ${msg.content}`);
    }
  }

  console.log("\n--- Final Output ---");
  const lastAI = result.messages.filter((m) => m._getType() === "ai").at(-1);
  console.log(lastAI?.content || "No response generated.");
  rl.close();
});
