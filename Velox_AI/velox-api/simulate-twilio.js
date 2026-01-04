// velox-api/simulate-twilio.js
const WebSocket = require('ws');

// Connect to your local server
const ws = new WebSocket('ws://localhost:8080/streams/voice');

ws.on('open', () => {
    console.log('🤖 Simulating Twilio Connection...');

    // 1. Send the "Start" event (Just like Twilio does)
    const startMsg = {
        event: "start",
        start: {
            streamSid: "stream_12345",
            callSid: "call_simulation_123",
            customParameters: {
                agentId: "support_bot_01"
            }
        }
    };
    ws.send(JSON.stringify(startMsg));

    // 2. Wait a second, then close
    setTimeout(() => {
        console.log('🤖 Simulation ending...');
        ws.close();
    }, 2000);
});