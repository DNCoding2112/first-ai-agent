import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { MessagesAnnotation, StateGraph } from "@langchain/langgraph";
import { SystemMessage, ToolMessage } from "@langchain/core/messages";
import { config } from "dotenv";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";

config();

// Define LLM
const llm = new ChatGoogleGenerativeAI({
  apiKey: process.env.GEMINI_API_KEY,
  model: "gemini-2.0-flash", // Use the desired Gemini model
});

// Define tools
const multiply = tool(
  async ({ a, b }) => {
    return `${a * b}`;
  },
  {
    name: "multiply",
    description: "Multiply two numbers together",
    schema: z.object({
      a: z.number().describe("first number"),
      b: z.number().describe("second number"),
    }),
  }
);

const add = tool(
  async ({ a, b }) => {
    return `${a + b}`;
  },
  {
    name: "add",
    description: "Add two numbers together",
    schema: z.object({
      a: z.number().describe("first number"),
      b: z.number().describe("second number"),
    }),
  }
);

const subtract = tool(
  async ({ a, b }) => {
    return `${a - b}`;
  },
  {
    name: "subtract",
    description: "Subtract two numbers",
    schema: z.object({
      a: z.number().describe("first number"),
      b: z.number().describe("second number"),
    }),
  }
);


const divide = tool(
  async ({ a, b }) => {
    return `${a / b}`;
  },
  {
    name: "divide",
    description: "Divide two numbers",
    schema: z.object({
      a: z.number().describe("first number"),
      b: z.number().describe("second number"),
    }),
  }
);

const mod = tool(
  async ({ a, b }) => `${a % b}`,
  {
    name: "mod",
    description: "Find the remainder when a is divided by b",
    schema: z.object({
      a: z.number().describe("dividend"),
      b: z.number().describe("divisor"),
    }),
  }
);

const power = tool(
  async ({ a, b }) => `${a ** b}`,
  {
    name: "power",
    description: "Raise a to the power of b",
    schema: z.object({
      a: z.number().describe("the base"),
      b: z.number().describe("the exponent"),
    }),
  }
);

const sqrt = tool(
  async ({ a }) => `${Math.sqrt(a)}`,
  {
    name: "sqrt",
    description: "Square root of a number",
    schema: z.object({
      a: z.number().describe("the number to take square root of"),
    }),
  }
);

const abs = tool(
  async ({ a }) => `${Math.abs(a)}`,
  {
    name: "abs",
    description: "Absolute value of a number",
    schema: z.object({
      a: z.number().describe("the number"),
    }),
  }
);

const sin = tool(
  async ({ a }) => `${Math.sin(a)}`,
  {
    name: "sin",
    description: "Sine of a (in radians)",
    schema: z.object({
      a: z.number().describe("the angle in radians"),
    }),
  }
);

const cos = tool(
  async ({ a }) => `${Math.cos(a)}`,
  {
    name: "cos",
    description: "Cosine of a (in radians)",
    schema: z.object({
      a: z.number().describe("the angle in radians"),
    }),
  }
);

const tan = tool(
  async ({ a }) => `${Math.tan(a)}`,
  {
    name: "tan",
    description: "Tangent of a (in radians)",
    schema: z.object({
      a: z.number().describe("the angle in radians"),
    }),
  }
);

const arcsin = tool(
  async ({ a }) => `${Math.asin(a)}`,
  {
    name: "arcsin",
    description: "Arcsine (inverse sine) of a, returns radians",
    schema: z.object({
      a: z.number().describe("the value between -1 and 1"),
    }),
  }
);

const arccos = tool(
  async ({ a }) => `${Math.acos(a)}`,
  {
    name: "arccos",
    description: "Arccosine (inverse cosine) of a, returns radians",
    schema: z.object({
      a: z.number().describe("the value between -1 and 1"),
    }),
  }
);

const arctan = tool(
  async ({ a }) => `${Math.atan(a)}`,
  {
    name: "arctan",
    description: "Arctangent (inverse tangent) of a, returns radians",
    schema: z.object({
      a: z.number().describe("the value"),
    }),
  }
);

const log = tool(
  async ({ a }) => `${Math.log(a)}`,
  {
    name: "log",
    description: "Natural logarithm (base e) of a",
    schema: z.object({
      a: z.number().describe("the number (> 0)"),
    }),
  }
);

const log10 = tool(
  async ({ a }) => `${Math.log10(a)}`,
  {
    name: "log10",
    description: "Base 10 logarithm of a",
    schema: z.object({
      a: z.number().describe("the number (> 0)"),
    }),
  }
);

const exp = tool(
  async ({ a }) => `${Math.exp(a)}`,
  {
    name: "exp",
    description: "e raised to the power of a",
    schema: z.object({
      a: z.number().describe("the exponent"),
    }),
  }
);

const round = tool(
  async ({ a }) => `${Math.round(a)}`,
  {
    name: "round",
    description: "Round a to the nearest integer",
    schema: z.object({
      a: z.number().describe("the number"),
    }),
  }
);

const floor = tool(
  async ({ a }) => `${Math.floor(a)}`,
  {
    name: "floor",
    description: "Round a down to the nearest integer",
    schema: z.object({
      a: z.number().describe("the number"),
    }),
  }
);

const ceil = tool(
  async ({ a }) => `${Math.ceil(a)}`,
  {
    name: "ceil",
    description: "Round a up to the nearest integer",
    schema: z.object({
      a: z.number().describe("the number"),
    }),
  }
);

const max = tool(
  async ({ a, b }) => `${Math.max(a, b)}`,
  {
    name: "max",
    description: "Return the larger of two numbers",
    schema: z.object({
      a: z.number().describe("first number"),
      b: z.number().describe("second number"),
    }),
  }
);

const min = tool(
  async ({ a, b }) => `${Math.min(a, b)}`,
  {
    name: "min",
    description: "Return the smaller of two numbers",
    schema: z.object({
      a: z.number().describe("first number"),
      b: z.number().describe("second number"),
    }),
  }
);

// Augment the LLM with tools
const tools = [
  add,
  subtract,
  multiply,
  divide,
  mod,
  power,
  sqrt,
  abs,
  sin,
  cos,
  tan,
  arcsin,
  arccos,
  arctan,
  log,
  log10,
  exp,
  round,
  floor,
  ceil,
  max,
  min
];
const toolsByName = Object.fromEntries(tools.map((tool) => [tool.name, tool]));
const llmWithTools = llm.bindTools(tools);

async function llmCall(state) {
  const result = await llmWithTools.invoke([
    {
      role: "system",
      content: "You are a helpful assistant tasked with performing arithmetic on a set of inputs.",
    },
    ...state.messages,
  ]);

  return {
    messages: [result],
  };
}

async function ToolNode(state) {
  const results = [];
  const lastMessage = state.messages.at(-1);
  if (lastMessage?.tool_calls?.length) {
    for (const toolCall of lastMessage.tool_calls) {
      const tool = toolsByName[toolCall.name];
      const observation = await tool.invoke(toolCall.args);
      results.push(
        new ToolMessage({
          content: observation,
          tool_call_id: toolCall.id,
        })
      );
    }
  }
  return { messages: results };
}

function shouldContinue(state) {
  const messages = state.messages;
  const lastMessage = messages.at(-1);

  if (lastMessage?.tool_calls?.length) {
    return "Action";
  }
  return "__end__";
}

const toolNode = ToolNode;

const agentBuilder = new StateGraph(MessagesAnnotation)
  .addNode("llmCall", llmCall)
  .addNode("tools", toolNode)
  .addEdge("__start__", "llmCall")
  .addConditionalEdges("llmCall", shouldContinue, {
    Action: "tools",
    __end__: "__end__",
  })
  .addEdge("tools", "llmCall")
  .compile();

// const messages = [
//   {
//     role: "user",
//     content: " Add 3 and 4. Then multiply the result by 10. Finally, divide by 2 and subtract 5.",
//   },
// ];



// const result = await agentBuilder.invoke({ messages });
// console.log(result.messages);

import readline from "readline";

// Create simple console input
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

rl.question("Enter your query: ", async (userInput) => {
  const messages = [
    {
      role: "user",
      content: userInput
    }
  ];

  const result = await agentBuilder.invoke({ messages });

  // Flatten messages into clean strings
  console.log("\n--- Conversation Trace ---");
  for (const msg of result.messages) {
    if (msg._getType() === "human") {
      console.log(`User: ${msg.content}`);
    } else if (msg._getType() === "ai") {
      if (msg.tool_calls?.length) {
        for (const call of msg.tool_calls) {
          console.log(`AI decided: Call tool "${call.name}" with args ${JSON.stringify(call.args)}`);
        }
      } else {
        console.log(`AI: ${typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content)}`);
      }
    } else if (msg._getType() === "tool") {
      console.log(`Tool Result: ${msg.content}`);
    }
  }

  console.log("\n--- Final Answer ---");
  const lastAI = result.messages.filter(m => m._getType() === "ai").at(-1);
  console.log(lastAI.content);

  rl.close();
});
