// import { tool } from "@langchain/core/tools";
// import { z } from "zod";
// import { MessagesAnnotation, StateGraph } from "@langchain/langgraph";
// // import { ToolNode } from "@langchain/langgraph/prebuilt";
// import {
//   SystemMessage,
//   ToolMessage
// } from "@langchain/core/messages";
// import {config } from "dotenv";
// import { ChatOpenAI } from "@langchain/openai";

// config();

// //define LLM
// const llm = new ChatOpenAI({
//   apiKey: process.env.OPENAI_API_KEY,
//   modelName: "gpt-3.5-turbo",
// });


// // Define tools
// const multiply = tool(
//   async ({ a, b }) => {
//     return `${a * b}`;
//   },
//   {
//     name: "multiply",
//     description: "Multiply two numbers together",
//     schema: z.object({
//       a: z.number().describe("first number"),
//       b: z.number().describe("second number"),
//     }),
//   }
// );

// const add = tool(
//   async ({ a, b }) => {
//     return `${a + b}`;
//   },
//   {
//     name: "add",
//     description: "Add two numbers together",
//     schema: z.object({
//       a: z.number().describe("first number"),
//       b: z.number().describe("second number"),
//     }),
//   }
// );

// const divide = tool(
//   async ({ a, b }) => {
//     return `${a / b}`;
//   },
//   {
//     name: "divide",
//     description: "Divide two numbers",
//     schema: z.object({
//       a: z.number().describe("first number"),
//       b: z.number().describe("second number"),
//     }),
//   }
// );

// // Augment the LLM with tools
// const tools = [add, multiply, divide];
// const toolsByName = Object.fromEntries(tools.map((tool) => [tool.name, tool]));
// const llmWithTools = llm.bindTools(tools);

// async function llmCall(state) {
//   // LLM decides whether to call a tool or not
//   const result = await llmWithTools.invoke([
//     {
//       role: "system",
//       content: "You are a helpful assistant tasked with performing arithmetic on a set of inputs."
//     },
//     ...state.messages
//   ]);

//   return {
//     messages: [result]
//   };
// }

// async function ToolNode(state){
//     const results = [];
//     const lastMessage = state.messages.at(-1);
//     if(lastMessage?.tool_calls?.length){
//         for(const toolCall of lastMessage.tool_calls){
//             const tool = toolsByName[toolCall.name];
//             const observation = await tool.invoke(toolCall.args);
//             results.push(new ToolMessage({
//                 // name: toolCall.name,
//                 content: observation,
//                 tool_call_id: toolCall.id
//             })); 
//         }
//     }
//     return { messages: results };
// }

// // Conditional edge function to route to the tool node or end
// function shouldContinue(state) {
//   const messages = state.messages;
//   const lastMessage = messages.at(-1);

//   // If the LLM makes a tool call, then perform an action
//   if (lastMessage?.tool_calls?.length) {
//     return "Action";
//   }
//   // Otherwise, we stop (reply to the user)
//   return "__end__";
// }

// const toolNode = ToolNode;

// const agentBuilder = new StateGraph(MessagesAnnotation)
//   .addNode("llmCall", llmCall)
//   .addNode("tools", toolNode)
//   // Add edges to connect nodes
//   .addEdge("__start__", "llmCall")
//   .addConditionalEdges(
//     "llmCall",
//     shouldContinue,
//     {
//       // Name returned by shouldContinue : Name of next node to visit
//       "Action": "tools",
//       "__end__": "__end__",
//     }
//   )
//   .addEdge("tools", "llmCall")
//   .compile();

// // Invoke
// const messages = [{
//   role: "user",
//   content: "Add 3 and 4."
// }];
// const result = await agentBuilder.invoke({ messages });
// console.log(result.messages);

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

// Augment the LLM with tools
const tools = [add, multiply, divide];
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

const messages = [
  {
    role: "user",
    content: "Add 3 and 4 multiply by 10 and then divide by 2.",
  },
];

const result = await agentBuilder.invoke({ messages });
console.log(result.messages);
