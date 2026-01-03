import { Router } from "express";
import { validateTwilioWebhook } from "../middleware/twilioAuth";
import VoiceResponse from "twilio/lib/twiml/VoiceResponse";

const router = Router();

router.post("/incoming", validateTwilioWebhook, (req, res) => {
  const twiml = new VoiceResponse();
  
  // The <Connect> verb upgrades HTTP -> WebSocket
  const connect = twiml.connect();
  const stream = connect.stream({
    url: `wss://${req.headers.host}/streams/voice`, // Uses the same host/ngrok
  });
  
  // Optional: Pass custom data to the stream (like which Agent to use)
  stream.parameter({ name: "agentId", value: "support_bot_01" });

  res.type("text/xml");
  res.send(twiml.toString());
});

export default router;