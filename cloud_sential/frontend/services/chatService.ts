import { db } from "../src/firebase"; // Adjust path if needed
import { collection, addDoc, serverTimestamp, doc, updateDoc } from "firebase/firestore";

// --- Types ---
interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
  text?: string;
}

// New Interface for the AI Return Type
interface AIResult {
  response: string;
  logs: any[];
}

/**
 * 1. SEND MESSAGE TO FIRESTORE
 * UPDATED: Accepts optional 'logs' array (5th argument)
 */
export const sendMessageToFirestore = async (
  userId: string, 
  chatId: string | null, 
  message: string, 
  role: "user" | "assistant" | "system" = "user",
  logs: any[] = []
): Promise<string> => {
  try {
    let currentChatId = chatId;

    // SCENARIO A: Brand new chat
    if (!currentChatId) {
      const newChatRef = await addDoc(collection(db, "chats"), {
        userId: userId,
        title: message.slice(0, 30) + "...",
        createdAt: serverTimestamp(),
        lastUpdatedAt: serverTimestamp()
      });
      currentChatId = newChatRef.id;
    }

    // SCENARIO B: Add message to sub-collection
    const messagesRef = collection(db, "chats", currentChatId, "messages");
    
    // Construct the payload
    const messagePayload: any = {
      text: message,
      role: role,
      createdAt: serverTimestamp(),
    };

    // Only save logs to DB if they exist (saves space)
    if (logs && logs.length > 0) {
      messagePayload.logs = logs; 
    }
    
    await addDoc(messagesRef, messagePayload);

    // SCENARIO C: Update parent timestamp
    const chatRef = doc(db, "chats", currentChatId);
    await updateDoc(chatRef, { lastUpdatedAt: serverTimestamp() });

    return currentChatId;
  } catch (error) {
    console.error("Error sending message to Firestore:", error);
    throw error;
  }
};

/**
 * 2. GET AI RESPONSE
 * UPDATED: Returns { response, logs } object instead of just string
 */
export const getAIResponse = async (currentMessage: string, history: ChatMessage[] = []): Promise<AIResult> => {
  try {
    // 1. SANITIZE HISTORY
    const cleanHistory = history.map(msg => {
      let cleanContent = msg.content || msg.text || "";
      
      if (typeof cleanContent === 'object' && cleanContent !== null) {
        // @ts-ignore
        cleanContent = cleanContent.response || cleanContent.answer || JSON.stringify(cleanContent);
      }

      return {
        role: msg.role,
        content: String(cleanContent)
      };
    });

    console.log("🚀 SENDING CLEAN PAYLOAD:", JSON.stringify({
      message: currentMessage,
      history: cleanHistory
    }, null, 2));

    // 2. CALL BACKEND
    const response = await fetch('http://127.0.0.1:8000/chat', { 
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: currentMessage,
        history: cleanHistory
      })
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`❌ Backend Error (${response.status}):`, errorText);
      throw new Error(`Backend Error: ${response.statusText}`);
    }

    // 3. PARSE RESPONSE
    const data = await response.json();
    
    // UPDATED RETURN: Returns the Object structure your calling code expects
    return {
      response: data.response || data.answer || "No response received.",
      logs: data.logs || [] // Pass logs through from Backend
    };

  } catch (error) {
    console.error("API Request Failed:", error);
    // Return a safe fallback object on error
    return {
      response: "Target system unresponsive. Ensure Backend is running on Port 8000.",
      logs: [] 
    };
  }
};