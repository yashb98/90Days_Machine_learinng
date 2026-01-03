"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = require("express");
const twilioAuth_1 = require("../middleware/twilioAuth");
const VoiceResponse_1 = __importDefault(require("twilio/lib/twiml/VoiceResponse"));
const router = (0, express_1.Router)();
router.post("/incoming", twilioAuth_1.validateTwilioWebhook, (req, res) => {
    const twiml = new VoiceResponse_1.default();
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
exports.default = router;
