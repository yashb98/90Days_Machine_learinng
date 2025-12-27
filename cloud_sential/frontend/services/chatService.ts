import { db } from "../src/firebase"; // Adjust path to your firebase.ts
import { collection, addDoc, serverTimestamp, doc, updateDoc } from "firebase/firestore";

// Define Types
export interface ChatMessage {
  text: string;
  role: "user" | "assistant" | "system";
  createdAt: any;
}

/**
 * Sends a message to Firestore.
 * - Creates a new chat document if chatId is null.
 * - Adds message to sub-collection.
 * - Updates parent 'lastUpdatedAt'.
 */
export const sendMessageToFirestore = async (
  userId: string, 
  chatId: string | null, 
  message: string, 
  role: "user" | "assistant" | "system" = "user"
): Promise<string> => {
  try {
    let currentChatId = chatId;

    // 1. Create new Chat Document if it doesn't exist
    if (!currentChatId) {
      const newChatRef = await addDoc(collection(db, "chats"), {
        userId: userId,
        title: message.slice(0, 30) + "...", 
        createdAt: serverTimestamp(),
        lastUpdatedAt: serverTimestamp()
      });
      currentChatId = newChatRef.id;
    }

    // 2. Add Message to Sub-collection
    const messagesRef = collection(db, "chats", currentChatId, "messages");
    await addDoc(messagesRef, {
      text: message,
      role: role,
      createdAt: serverTimestamp(),
    });

    // 3. Update Parent Timestamp (for sidebar sorting)
    const chatRef = doc(db, "chats", currentChatId);
    await updateDoc(chatRef, { lastUpdatedAt: serverTimestamp() });

    return currentChatId;
  } catch (error) {
    console.error("Error sending message:", error);
    throw error;
  }
};

/**
 * Simulated AI Response.
 * Replace this with your actual API call to Gemini/OpenAI.
 */
export const getAIResponse = async (userText: string): Promise<string> => {
  await new Promise(resolve => setTimeout(resolve, 1000)); // Fake delay
  return `Analyzed: "${userText}". Systems nominal.`;
};