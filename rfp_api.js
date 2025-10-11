import express from "express";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { MessagesAnnotation, StateGraph } from "@langchain/langgraph";
import { ToolMessage } from "@langchain/core/messages";
import { config } from "dotenv";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";

config();

const app = express();
app.use(express.json());

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
// Master Agent Node
// -------------------------
async function MasterAgent(state) {
  const result = await llmWithTools.invoke(
    [
      {
        role: "system",
        content: `
You are the Master Agent. Decide freely which sub-agent (SalesAgent, TechnicalAgent, PricingAgent) 
should be called based on the user's query. Only call an agent if needed.
Once you have enough information, provide a consolidated RFP response.
        `,
      },
      ...state.messages,
    ],
    { configurable: { run_name: "MasterAgentLLM", tags: ["rfp", "orchestrator", "langsmith"] } }
  );
  return { messages: [result] };
}

// -------------------------
// Tool Executor Node
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
// Flow Control
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
// API Endpoint
// -------------------------
app.post("/rfp-query", async (req, res) => {
  const { query } = req.body;
  if (!query) return res.status(400).json({ error: "Missing query in request body." });

  try {
    const messages = [{ role: "user", content: query }];

    const result = await RFPGraph.invoke(
      { messages },
      { configurable: { run_name: "RFPOrchestratorRun", tags: ["rfp", "multiagent", "langsmith"] } }
    );

    const trace = result.messages.map((msg) => {
      if (msg._getType() === "human") return { role: "user", content: msg.content };
      if (msg._getType() === "ai") {
        return msg.tool_calls?.length
          ? { role: "masterAgent", toolCalls: msg.tool_calls }
          : { role: "masterAgent", content: msg.content };
      }
      if (msg._getType() === "tool") return { role: "agent", content: msg.content };
      return { role: "unknown", content: msg.content };
    });

    const finalAI = result.messages.filter((m) => m._getType() === "ai").at(-1);

    res.json({
      conversationTrace: trace,
      finalOutput: finalAI?.content || "No response generated.",
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to process RFP query." });
  }
});

// -------------------------
// Start Server
// -------------------------
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`RFP Agent API running on port ${PORT}`);
});
