// server.js
import * as dotenv from 'dotenv';
dotenv.config();

import express from 'express';
import bodyParser from 'body-parser';
import path from 'path';
import { fileURLToPath } from 'url';

import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';
import { GoogleGenerativeAI } from '@google/generative-ai';

// Path setup
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PORT = process.env.PORT || 3000;

// Env warnings
if (!process.env.GEMINI_API_KEY) console.warn('⚠️ Missing GEMINI_API_KEY in .env');
if (!process.env.PINECONE_INDEX_NAME) console.warn('⚠️ Missing PINECONE_INDEX_NAME in .env');

const app = express();
app.use(bodyParser.json({ limit: '5mb' }));
app.use(express.static(path.join(__dirname, 'public')));

// Gemini Client
const ai = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// 🧩 Step 1: Rewrite user question
async function transformQuery(question, history = []) {
  const model = ai.getGenerativeModel({ model: 'gemini-2.0-flash' });

  const prompt = `
You are a query rewriting assistant.
Given the chat history and the follow-up question, rewrite the follow-up question into a complete, standalone question.
Return only the rewritten question.

Chat History:
${history.map(h => h.text).join('\n')}

Follow-up Question:
${question}
`;

  // ✅ FIX: Correct structure for Gemini SDK
  const response = await model.generateContent({
    contents: [
      {
        role: 'user',
        parts: [{ text: prompt }],
      },
    ],
  });

  return response.response.text().trim();
}

// 🧠 Step 2: Ask route — rewrite → embed → Pinecone → ask Gemini
app.post('/ask', async (req, res) => {
  try {
    const { question } = req.body;
    if (!question?.trim()) {
      return res.status(400).json({ error: 'question is required' });
    }

    // Rewrite question
    const rewritten = await transformQuery(question);

    // Embeddings
    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GEMINI_API_KEY,
      model: 'text-embedding-004',
    });

    const queryVector = await embeddings.embedQuery(rewritten);

    // Pinecone Search
    const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
    const index = pinecone.Index(process.env.PINECONE_INDEX_NAME);

    const searchResults = await index.query({
      vector: queryVector,
      topK: 6,
      includeMetadata: true,
    });

    const context = (searchResults.matches || [])
      .map(match => match.metadata?.text || '')
      .filter(Boolean)
      .join('\n\n---\n\n');

    // Gemini query — only 'user' role used
    const model = ai.getGenerativeModel({ model: 'gemini-2.5-flash' });

    const prompt = `
You are a helpful Data Structures and Algorithms assistant.
Answer the user's question based ONLY on the provided context.
If the answer cannot be found, say: "I could not find the answer in the provided document."

Context:
${context || '(No relevant context found in the database.)'}

User Question:
${question}
`;

    // ✅ FIX: Use proper request format for Gemini SDK
    const response = await model.generateContent({
      contents: [
        {
          role: 'user',
          parts: [{ text: prompt }],
        },
      ],
    });

    const answer = response.response.text().trim();
    res.json({ answer, context });
  } catch (err) {
    console.error('❌ /ask Error:', err);
    res.status(500).json({ error: err.message });
  }
});

// 🧾 Step 3: PDF Indexing
app.post('/index', async (req, res) => {
  try {
    const PDF_PATH = './dsa.pdf';
    const pdfLoader = new PDFLoader(PDF_PATH);
    const docs = await pdfLoader.load();

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
    const splitDocs = await splitter.splitDocuments(docs);

    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GEMINI_API_KEY,
      model: 'text-embedding-004',
    });

    const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
    const index = pinecone.Index(process.env.PINECONE_INDEX_NAME);

    await PineconeStore.fromDocuments(splitDocs, embeddings, {
      pineconeIndex: index,
      maxConcurrency: 5,
    });

    res.json({ status: 'ok', message: 'PDF indexed to Pinecone successfully' });
  } catch (err) {
    console.error('❌ /index Error:', err);
    res.status(500).json({ error: err.message });
  }
});

// Serve frontend
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Start server
app.listen(PORT, () => console.log(`🚀 Server running on http://localhost:${PORT}`));
