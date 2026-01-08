import WebSocket from "ws";
import { randomUUID } from "crypto"; // Use Node's built-in crypto instead of external 'uuid'

// ⚙️ CONFIGURATION
const WS_URL = "ws://localhost:8080/streams";
const NUM_CLIENTS = 5; // Start low to test safely
const DURATION_MS = 10000; // 10 seconds per call

const log = (id: number, msg: string) => console.log(`[Bot ${id}] ${msg}`);

const runClient = (id: number) => {
  const ws = new WebSocket(WS_URL);
  const streamSid = randomUUID(); // Built-in Node function
  const callSid = `sim_call_${id}_${Date.now()}`;
  let audioInterval: NodeJS.Timeout;

  ws.on("open", () => {
    log(id, "Connected");

    // 1. Send Start Event (Mimic Twilio)
    ws.send(JSON.stringify({
      event: "start",
      start: {
        streamSid,
        callSid,
        customParameters: { agentId: "load_tester" }
      }
    }));

    // 2. Stream Audio (Mimic Speaking)
    // We send random bytes (static noise) to trigger data flow
    audioInterval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        // Generate 160 bytes of random noise (approx 20ms of audio)
        const noise = Buffer.alloc(160);
        for (let i = 0; i < 160; i++) noise[i] = Math.floor(Math.random() * 255);

        ws.send(JSON.stringify({
          event: "media",
          streamSid,
          media: {
            payload: noise.toString("base64")
          }
        }));
      }
    }, 20); // Send every 20ms

    // 3. Stop after DURATION_MS
    setTimeout(() => {
      log(id, "Hanging up");
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ event: "stop" }));
        ws.close();
      }
      clearInterval(audioInterval);
    }, DURATION_MS);
  });

  ws.on("message", (data) => {
    try {
      const msg = JSON.parse(data.toString());
      if (msg.event === "media") {
        log(id, " Heard Audio Response (TTS works)");
      }
      if (msg.event === "clear") {
        log(id, "Received Interrupt Signal");
      }
    } catch (e) {
      // Ignore parse errors from raw audio
    }
  });

  ws.on("error", (e) => log(id, `Error: ${e.message}`));
  ws.on("close", () => log(id, "Disconnected"));
};

// 🚀 LAUNCH THE SWARM
console.log(`Launching ${NUM_CLIENTS} concurrent calls...`);
for (let i = 0; i < NUM_CLIENTS; i++) {
  // Stagger them slightly so they don't hit exactly at the same millisecond
  setTimeout(() => runClient(i + 1), i * 200);
}