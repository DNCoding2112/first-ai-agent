import { GoogleGenerativeAI } from '@google/generative-ai';
import dotenv from 'dotenv';

// Load environment variables from .env file
dotenv.config();

const apiKey = process.env.GEMINI_API_KEY;

const genAI = new GoogleGenerativeAI(apiKey);
const model = genAI.getGenerativeModel({ model: 'gemini-2.0-flash' });

async function makeGeminiRequest() {
  try {
    const result = await model.generateContent('Hello, Gemini!');
    console.log('Gemini Response:', result.response.text());
  } catch (error) {
    console.error('Error:', error.message);
  }
}

makeGeminiRequest();